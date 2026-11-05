"""
models.py — residual U-Net for ring artifact removal.

Design choices
--------------
* Residual (artifact) learning. In the log/line-integral domain a gain error
  is an additive offset, and FBP is linear, so to first order

        noisy = clean + R,     R = FBP(stripe pattern)

  The network predicts R and returns  restored = noisy - s * f(noisy).  R is
  sparse, zero-mean and structurally simple (concentric circles), which is
  easier to regress than the full image. The output conv is initialised
  near zero, so training starts from (almost) the identity map.

* Residual output scale s (default 0.02). Ring amplitudes are ~1e-3–1e-2 in
  normalised units. Adam's per-parameter step is ~lr regardless of gradient
  magnitude, so an unscaled head that must produce O(1e-3) outputs is
  updated by steps far larger than the target and dithers around the
  identity (observed: no improvement over the input even when overfitting
  one batch). Scaling the head output by s lets f operate in O(0.1–1) units.

* Large receptive field. A ring is a global structure: telling a ring from a
  circular anatomical edge needs context along its circumference. With depth
  4 (16x down-sampling) and two 3x3 convs per residual block, the bottleneck
  receptive field exceeds 200 px, comparable to a 256x256 slice.

* Centre awareness (optional, default on). Rings are centred on the rotation
  axis, i.e. the image centre. Two extra input channels give each pixel its
  normalised radius r and a constant 1 (CoordConv-style), letting
  translation-equivariant convolutions condition on distance to the centre.

* GroupNorm instead of BatchNorm: stable with the small batches typical for
  medical images.

Input  : (B, 1, H, W) in [0, 1]. Any H, W (padded internally to a multiple
         of 2**depth and cropped back).
Output : (B, 1, H, W) restored image.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    groups = 8 if ch % 8 == 0 else 1
    return nn.GroupNorm(groups, ch)


class ResBlock(nn.Module):
    """Pre-activation residual block: (GN-SiLU-Conv) x2 + projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            _gn(in_ch), nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.body(x) + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_blocks: int, dropout: float):
        super().__init__()
        self.pool = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)  # learned downsample
        blocks = [ResBlock(in_ch, out_ch, dropout)]
        blocks += [ResBlock(out_ch, out_ch, dropout) for _ in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_blocks: int, dropout: float):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear",
                                            align_corners=False),
                                nn.Conv2d(in_ch, in_ch // 2, 3, padding=1))
        blocks = [ResBlock(in_ch // 2 + skip_ch, out_ch, dropout)]
        blocks += [ResBlock(out_ch, out_ch, dropout) for _ in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, skip):
        x = self.up(x)
        return self.blocks(torch.cat([x, skip], dim=1))


class RingRemovalUNet(nn.Module):
    """Residual U-Net: restored = input - predicted_artifact."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32,
                 depth: int = 4, blocks_per_level: int = 2, dropout: float = 0.0,
                 coord_channels: bool = True, residual: bool = True,
                 residual_scale: float = 0.02):
        super().__init__()
        self.depth = depth
        self.coord = coord_channels
        self.residual = residual
        self.residual_scale = residual_scale if residual else 1.0
        stem_in = in_channels + (2 if coord_channels else 0)

        chs = [base * 2 ** i for i in range(depth + 1)]       # e.g. 32..512
        self.stem = nn.Conv2d(stem_in, chs[0], 3, padding=1)
        self.enc0 = nn.Sequential(*[ResBlock(chs[0], chs[0], dropout)
                                    for _ in range(blocks_per_level)])
        self.downs = nn.ModuleList([Down(chs[i], chs[i + 1], blocks_per_level, dropout)
                                    for i in range(depth)])
        self.mid = nn.Sequential(ResBlock(chs[-1], chs[-1], dropout),
                                 ResBlock(chs[-1], chs[-1], dropout))
        self.ups = nn.ModuleList([Up(chs[i + 1], chs[i], chs[i], blocks_per_level, dropout)
                                  for i in reversed(range(depth))])
        self.head = nn.Sequential(_gn(chs[0]), nn.SiLU(inplace=True),
                                  nn.Conv2d(chs[0], out_channels, 3, padding=1))
        # Near-zero init of the output conv: the untrained model is ~identity.
        # (Exactly zero weights block all upstream gradients at step 0 and, in
        # testing, made L1-type losses stall; a tiny random init avoids that.)
        nn.init.normal_(self.head[-1].weight, 0.0, 1e-4)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _coords(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        r = torch.sqrt(xx ** 2 + yy ** 2) / (2 ** 0.5)          # in [0, 1]
        coords = torch.stack([r, torch.ones_like(r)], 0)
        return coords.unsqueeze(0).expand(b, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (m - h % m) % m, (m - w % m) % m
        # symmetric padding keeps the image centre (ring centre) at the centre
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        xp = F.pad(x, pad, mode="reflect") if (ph or pw) else x

        z = torch.cat([xp, self._coords(xp)], 1) if self.coord else xp
        z = self.enc0(self.stem(z))
        skips = [z]
        for down in self.downs:
            z = down(z)
            skips.append(z)
        z = self.mid(skips.pop())
        for up in self.ups:
            z = up(z, skips.pop())
        out = self.head(z) * self.residual_scale

        if ph or pw:
            out = out[..., pad[2]:pad[2] + h, pad[0]:pad[0] + w]
        return x - out if self.residual else out


def build_model(name: str = "unet", **kw) -> nn.Module:
    if name == "unet":
        return RingRemovalUNet(**kw)
    if name == "unet_small":
        kw.setdefault("base", 16)
        kw.setdefault("depth", 4)
        kw.setdefault("blocks_per_level", 1)
        return RingRemovalUNet(**kw)
    raise ValueError(f"Unknown model '{name}'")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in ("unet", "unet_small"):
        m = build_model(name)
        x = torch.rand(2, 1, 250, 250)
        y = m(x)
        print(name, tuple(y.shape), f"{count_parameters(m) / 1e6:.2f} M params",
              f"max |y - x| at init: {(y - x).abs().max():.2e}")
