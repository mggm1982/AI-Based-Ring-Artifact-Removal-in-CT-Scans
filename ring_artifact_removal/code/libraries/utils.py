"""
utils.py — losses, metrics, visualisation and small helpers.

Metrics (all on images normalised to [0, 1], data_range = 1)
-------------------------------------------------------------
PSNR   = 10 log10( 1 / MSE )

SSIM   (Wang et al., 2004), Gaussian window sigma = 1.5, size 11:
         SSIM(x, y) = mean over windows of
            (2 mu_x mu_y + C1)(2 sigma_xy + C2) /
            ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)),
         C1 = (0.01 L)^2, C2 = (0.03 L)^2, L = 1.

Ring residual (RRE) — a ring-specific metric. PSNR/SSIM average over the
whole image and cannot tell ring residue from blur. Rings are radially
structured and angularly constant, so the angular mean of the error
e = y_hat - y over each 1-pixel annulus isolates them:

        e_bar(r) = mean_{theta} e(r, theta)
        RRE      = sqrt( mean_r e_bar(r)^2 )        (r inside the FOV circle)

Angular averaging over N_r pixels suppresses non-ring error by roughly
1/sqrt(N_r) while keeping ring error intact. Reported in normalised units.
"""

from __future__ import annotations

import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reproducibility / bookkeeping
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AverageMeter:
    def __init__(self):
        self.sum, self.n = 0.0, 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.n += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)


# ---------------------------------------------------------------------------
# SSIM / PSNR (differentiable, batched)
# ---------------------------------------------------------------------------

def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]          # (1, 1, k, k)


def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0,
         window_size: int = 11, sigma: float = 1.5, reduction: str = "mean"):
    """SSIM for (B, C, H, W). reduction: 'mean' | 'none' (per image)."""
    c = x.shape[1]
    win = _gaussian_window(window_size, sigma, x.device, x.dtype).expand(c, 1, -1, -1)
    pad = window_size // 2
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    def filt(t):
        return F.conv2d(F.pad(t, [pad] * 4, mode="reflect"), win, groups=c)

    mu_x, mu_y = filt(x), filt(y)
    sxx = filt(x * x) - mu_x ** 2
    syy = filt(y * y) - mu_y ** 2
    sxy = filt(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sxy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2)
    smap = num / den
    per_img = smap.flatten(1).mean(1)
    return per_img.mean() if reduction == "mean" else per_img


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0,
         reduction: str = "mean", eps: float = 1e-10):
    mse = ((x - y) ** 2).flatten(1).mean(1)
    val = 10.0 * torch.log10(data_range ** 2 / (mse + eps))
    return val.mean() if reduction == "mean" else val


def ring_residual_error(x: torch.Tensor, y: torch.Tensor,
                        reduction: str = "mean") -> torch.Tensor:
    """RMS over radius of the angular-mean error (see module docstring)."""
    e = (x - y)[:, 0]                                        # (B, H, W)
    b, h, w = e.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=e.device),
                            torch.arange(w, device=e.device), indexing="ij")
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(h, w) / 2 - 1
    ridx = torch.round(r).long()
    inside = r <= rmax
    ridx = ridx[inside]                                      # (P,)
    n_bins = int(ridx.max().item()) + 1
    counts = torch.bincount(ridx, minlength=n_bins).clamp_min(1).to(e.dtype)

    vals = e[:, inside]                                      # (B, P)
    sums = torch.zeros(b, n_bins, device=e.device, dtype=e.dtype)
    sums.scatter_add_(1, ridx.expand(b, -1), vals)
    prof = sums / counts                                     # (B, n_bins)
    per_img = torch.sqrt((prof[:, 1:] ** 2).mean(1))         # skip r=0 (1 px)
    return per_img.mean() if reduction == "mean" else per_img


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-image metrics, prediction clamped to [0, 1]."""
    pred = pred.clamp(0, 1).float()
    target = target.float()
    return {
        "psnr": psnr(pred, target, reduction="none"),
        "ssim": ssim(pred, target, reduction="none"),
        "mae": (pred - target).abs().flatten(1).mean(1),
        "rre": ring_residual_error(pred, target, reduction="none"),
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class L1SSIMLoss(nn.Module):
    """L = (1 - a) * L1 + a * (1 - SSIM)  [+ b * ring-profile loss].

    a = 0.84 follows Zhao, Gallo, Frosio & Kautz, "Loss Functions for Image
    Restoration with Neural Networks", IEEE Trans. Comput. Imaging 3(1), 2017
    (they used MS-SSIM; single-scale SSIM is used here for speed).

    The optional ring term penalises the angular-mean error profile directly,
    i.e. what the RRE metric measures; it is small but pushes the network to
    remove faint low-contrast rings that L1 under-weights.
    """

    def __init__(self, alpha: float = 0.84, ring_weight: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.ring_weight = ring_weight

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ls = 1.0 - ssim(pred, target)
        loss = (1 - self.alpha) * l1 + self.alpha * ls
        parts = {"l1": l1.detach(), "ssim_loss": ls.detach()}
        if self.ring_weight > 0:
            lr = ring_residual_error(pred, target)
            loss = loss + self.ring_weight * lr
            parts["ring_loss"] = lr.detach()
        return loss, parts


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_comparison(clean: np.ndarray, noisy: np.ndarray, restored: np.ndarray,
                    path: str, title: Optional[str] = None,
                    metrics_noisy: Optional[dict] = None,
                    metrics_restored: Optional[dict] = None,
                    diff_vmax: Optional[float] = None):
    """[Clean GT | Noisy input | Restored | |Restored - GT|] (+ |Noisy - GT| row)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean, noisy, restored = (np.squeeze(a) for a in (clean, noisy, restored))
    restored = np.clip(restored, 0, 1)
    d_rest = np.abs(restored - clean)
    d_noisy = np.abs(noisy - clean)
    vmax = diff_vmax or max(float(np.percentile(d_noisy, 99.5)), 1e-3)

    # display window: robust range of the clean image
    lo, hi = np.percentile(clean, [0.5, 99.5])
    if hi - lo < 1e-6:
        lo, hi = 0.0, 1.0

    def fmt(m):
        return "" if m is None else f"\nPSNR {m['psnr']:.2f} dB | SSIM {m['ssim']:.4f}"

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    panels = [(clean, "Clean (GT)", "gray", lo, hi),
              (noisy, "Noisy input" + fmt(metrics_noisy), "gray", lo, hi),
              (restored, "Restored" + fmt(metrics_restored), "gray", lo, hi),
              (d_rest, f"|Restored − GT|  (noisy: {d_noisy.mean():.4f} → "
                       f"{d_rest.mean():.4f} MAE)", "magma", 0, vmax)]
    for ax, (img, t, cmap, a, b) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=a, vmax=b)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history: list, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    if history and "val_loss" in history[0]:
        ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    if history and "val_psnr" in history[0]:
        ax[1].plot(ep, [h["val_psnr"] for h in history])
        ax[1].set_title("Validation PSNR (dB)"); ax[1].set_xlabel("epoch")
        ax[2].plot(ep, [h["val_ssim"] for h in history])
        ax[2].set_title("Validation SSIM"); ax[2].set_xlabel("epoch")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
