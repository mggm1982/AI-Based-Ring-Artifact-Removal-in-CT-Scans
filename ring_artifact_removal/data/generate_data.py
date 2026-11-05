#!/usr/bin/env python3
"""
generate_data.py — synthetic paired (ring-corrupted, clean) CT slices.

Physics of the simulation
-------------------------
Ring artifacts are not drawn in the image domain. They are produced the way a
real third-generation scanner produces them: a per-detector-channel calibration
error that is constant over the rotation, i.e. a vertical stripe in the
sinogram, which filtered back-projection (FBP) maps to a circle centred on the
rotation axis.

For detector channel i and view angle theta:

    ideal line integral         p(i, theta)   = ∫ mu(x) dl
    ideal intensity             I(i, theta)   = I0 * exp(-p)
    measured intensity          Im(i, theta)  = g_i(theta) * I(i, theta) + o_i
    flat-field-corrected value  pm(i, theta)  = -ln( Im / I0 )
                                              = -ln( g_i e^{-p} + o_i / I0 )

  * g_i  = 1 + eps_i : residual gain miscalibration (multiplicative). In the
           log domain this is an additive offset -ln(g_i) ≈ -eps_i that is
           independent of the object -> rings of constant amplitude.
  * o_i  : residual dark-current/offset error (additive in intensity). Its
           log-domain effect grows as e^{p}, so rings are stronger behind
           highly attenuating structures -> object-dependent rings.
  * optional linear drift of g_i over the scan -> rings of varying
           intensity around their circumference.

The "clean" image is FBP of the uncorrupted sinogram, with the SAME geometry,
filter and angular sampling. Hence noisy - clean contains only the ring
component (plus its FBP streak leakage); no FBP blur or angular undersampling
streaks are asked to be removed.

Output
------
    <out>/training/{noisy,clean}/XXXXX.npy   (80 %)
    <out>/testing/{noisy,clean}/XXXXX.npy    (20 %)
    <out>/dataset_meta.json                  normalisation window + settings
    <out>/ring_params.csv                    per-sample corruption parameters

.npy files hold float32 images in phantom (relative attenuation) units.
With --format png, images are first mapped through the global intensity
window to [0, 1] and saved as 16-bit PNG.

Example
-------
    python generate_data.py --num-images 1000 --size 256 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool

import numpy as np
from skimage.transform import iradon, radon

# ---------------------------------------------------------------------------
# Phantoms
# ---------------------------------------------------------------------------

# Modified Shepp–Logan (Toft 1996): A, a, b, x0, y0, phi[deg]
MODIFIED_SHEPP_LOGAN = np.array(
    [
        [1.00, 0.6900, 0.9200, 0.00, 0.0000, 0.0],
        [-0.80, 0.6624, 0.8740, 0.00, -0.0184, 0.0],
        [-0.20, 0.1100, 0.3100, 0.22, 0.0000, -18.0],
        [-0.20, 0.1600, 0.4100, -0.22, 0.0000, 18.0],
        [0.10, 0.2100, 0.2500, 0.00, 0.3500, 0.0],
        [0.10, 0.0460, 0.0460, 0.00, 0.1000, 0.0],
        [0.10, 0.0460, 0.0460, 0.00, -0.1000, 0.0],
        [0.10, 0.0460, 0.0230, -0.08, -0.6050, 0.0],
        [0.10, 0.0230, 0.0230, 0.00, -0.6060, 0.0],
        [0.10, 0.0230, 0.0460, 0.06, -0.6050, 0.0],
    ],
    dtype=np.float64,
)


def rasterize_ellipses(ellipses: np.ndarray, n: int) -> np.ndarray:
    """Sum of uniform ellipses on an n×n grid spanning [-1, 1]^2 (y up)."""
    coords = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(coords, coords[::-1])
    img = np.zeros((n, n), dtype=np.float64)
    for A, a, b, x0, y0, phi in ellipses:
        t = np.deg2rad(phi)
        c, s = np.cos(t), np.sin(t)
        xr = (xx - x0) * c + (yy - y0) * s
        yr = -(xx - x0) * s + (yy - y0) * c
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += A
    return img


def _global_transform(ell: np.ndarray, rng: np.random.Generator,
                      max_rot: float = 180.0, scale_rng=(0.85, 1.05)) -> np.ndarray:
    """Rotate/scale the whole phantom about the origin (keeps it in FOV)."""
    ell = ell.copy()
    rot = rng.uniform(-max_rot, max_rot)
    sc = rng.uniform(*scale_rng)
    t = np.deg2rad(rot)
    c, s = np.cos(t), np.sin(t)
    x0, y0 = ell[:, 3].copy(), ell[:, 4].copy()
    ell[:, 3] = sc * (c * x0 - s * y0)
    ell[:, 4] = sc * (s * x0 + c * y0)
    ell[:, 1:3] *= sc
    ell[:, 5] += rot
    return ell


def random_shepp_logan(n: int, rng: np.random.Generator) -> np.ndarray:
    """Modified Shepp–Logan with jittered geometry and contrast."""
    ell = MODIFIED_SHEPP_LOGAN.copy()
    inner = slice(2, None)
    ell[inner, 0] *= rng.uniform(0.6, 1.6, size=ell[inner].shape[0])
    ell[inner, 1:3] *= rng.uniform(0.8, 1.2, size=(ell[inner].shape[0], 2))
    ell[inner, 3:5] += rng.uniform(-0.04, 0.04, size=(ell[inner].shape[0], 2))
    ell[inner, 5] += rng.uniform(-15, 15, size=ell[inner].shape[0])
    ell[1, 0] = -rng.uniform(0.7, 0.85)          # skull/soft-tissue contrast
    ell = _global_transform(ell, rng)
    img = rasterize_ellipses(ell, n)
    return np.clip(img, 0.0, None)


def random_tissue_phantom(n: int, rng: np.random.Generator) -> np.ndarray:
    """Body outline + fat/soft tissue + organs + lesions + optional bone."""
    rows = []
    # body (skin/soft tissue) and a lower-attenuation fat-like inner layer
    a_body, b_body = rng.uniform(0.70, 0.88), rng.uniform(0.55, 0.85)
    body_mu = rng.uniform(0.45, 0.6)
    rows.append([body_mu, a_body, b_body, 0, 0, rng.uniform(-20, 20)])
    shrink = rng.uniform(0.88, 0.95)
    rows.append([-rng.uniform(0.05, 0.15), a_body * shrink, b_body * shrink,
                 0, 0, rows[0][5]])

    def inside(a, b):
        """Random centre such that an (a, b) ellipse stays inside the body."""
        r = rng.uniform(0, 1) ** 0.5
        ang = rng.uniform(0, 2 * np.pi)
        mx = max(a_body * shrink - a, 0.01)
        my = max(b_body * shrink - b, 0.01)
        return r * mx * np.cos(ang), r * my * np.sin(ang)

    # organs: moderate positive/negative contrast
    for _ in range(rng.integers(3, 8)):
        a, b = rng.uniform(0.08, 0.30), rng.uniform(0.08, 0.30)
        x0, y0 = inside(a, b)
        rows.append([rng.uniform(-0.15, 0.2), a, b, x0, y0, rng.uniform(0, 180)])
    # small lesions / vessels: low contrast, small size
    for _ in range(rng.integers(3, 12)):
        a, b = rng.uniform(0.01, 0.06), rng.uniform(0.01, 0.06)
        x0, y0 = inside(a, b)
        rows.append([rng.uniform(-0.1, 0.15), a, b, x0, y0, rng.uniform(0, 180)])
    # bone: few high-attenuation structures
    if rng.random() < 0.7:
        for _ in range(rng.integers(1, 4)):
            a, b = rng.uniform(0.03, 0.12), rng.uniform(0.03, 0.12)
            x0, y0 = inside(a, b)
            rows.append([rng.uniform(0.3, 0.5), a, b, x0, y0, rng.uniform(0, 180)])

    img = rasterize_ellipses(np.asarray(rows, dtype=np.float64), n)
    img = np.clip(img, 0.0, None)
    return img / max(img.max(), 1e-8)  # max attenuation -> 1


# ---------------------------------------------------------------------------
# Ring corruption in the sinogram domain
# ---------------------------------------------------------------------------

@dataclass
class RingParams:
    sigma_gain: float          # std of small gain errors on all channels
    n_defects: int             # number of strongly miscalibrated channel groups
    max_defect: float          # largest |eps| among defects
    sigma_offset: float        # std of offset error o_i / I0
    drift: bool                # gain drifts linearly over the scan
    mean_abs_eps: float        # mean |g_i - 1| over channels


def sample_detector_errors(n_det: int, n_views: int, rng: np.random.Generator,
                           strength: float = 1.0):
    """Returns gain map g (n_det, n_views), offset o (n_det, 1), params."""
    sigma_gain = strength * rng.uniform(0.001, 0.006)
    eps = rng.normal(0.0, sigma_gain, size=n_det)

    n_defects = int(rng.poisson(4.0 * strength))
    max_defect = 0.0
    for _ in range(n_defects):
        centre = rng.integers(int(0.03 * n_det), int(0.97 * n_det))
        width = int(rng.choice([1, 1, 1, 2, 3, 5]))
        amp = strength * rng.uniform(0.008, 0.04) * rng.choice([-1, 1])
        lo, hi = centre, min(centre + width, n_det)
        eps[lo:hi] += amp
        max_defect = max(max_defect, abs(amp))

    g = (1.0 + eps)[:, None] * np.ones((1, n_views))

    drift = bool(rng.random() < 0.3)
    if drift:  # per-channel linear drift over rotation -> partial-arc rings
        d = rng.normal(0.0, 0.5 * sigma_gain + 1e-4, size=n_det)
        t = np.linspace(-0.5, 0.5, n_views)[None, :]
        g = g * (1.0 + d[:, None] * t)

    sigma_offset = strength * rng.uniform(0.0, 3e-4)
    o = rng.normal(0.0, sigma_offset, size=(n_det, 1))

    params = RingParams(sigma_gain, n_defects, max_defect, sigma_offset, drift,
                        float(np.mean(np.abs(eps))))
    return g, o, params


def simulate_pair(phantom: np.ndarray, rng: np.random.Generator, n_views: int,
                  max_attenuation: float, strength: float,
                  scan_range: float = 360.0):
    """Forward-project, corrupt detector response, reconstruct both by FBP.

    With a 360-degree scan each detector channel at signed offset s from the
    rotation axis traces a full circle of radius |s| (as in clinical fan-beam
    scanners). A 180-degree parallel scan gives semicircles instead.
    """
    n = phantom.shape[0]
    theta = np.linspace(0.0, scan_range, n_views, endpoint=False)

    # radon sums pixel values along rays; one pixel = 2/n of the unit FOV.
    # Scale so that a path of length 2 (full diameter) through mu=1 gives
    # `max_attenuation` (a dimensionless line integral, ~4–5 for a body).
    scale = max_attenuation / n
    sino = radon(phantom, theta=theta, circle=True) * scale       # p >= 0

    g, o, params = sample_detector_errors(sino.shape[0], n_views, rng, strength)
    transmitted = np.exp(-sino)
    measured = g * transmitted + o
    measured = np.clip(measured, 1e-6, None)                       # avoid log(<=0)
    sino_corrupt = -np.log(measured)

    kw = dict(theta=theta, filter_name="ramp", circle=True, output_size=n)
    clean = iradon(sino, **kw) / scale
    noisy = iradon(sino_corrupt, **kw) / scale
    return clean.astype(np.float32), noisy.astype(np.float32), params


# ---------------------------------------------------------------------------
# Worker / IO
# ---------------------------------------------------------------------------

def _work(args):
    idx, seed, cfg = args
    rng = np.random.default_rng(seed)
    if rng.random() < cfg["sl_fraction"]:
        kind, ph = "shepp_logan", random_shepp_logan(cfg["size"], rng)
    else:
        kind, ph = "tissue", random_tissue_phantom(cfg["size"], rng)
    clean, noisy, params = simulate_pair(ph, rng, cfg["n_views"],
                                         cfg["max_attenuation"], cfg["strength"],
                                         cfg["scan_range"])
    return idx, kind, clean, noisy, params


def to_png16(img: np.ndarray, lo: float, hi: float, path: str):
    from PIL import Image
    x = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    Image.fromarray((x * 65535.0 + 0.5).astype(np.uint16)).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)),
                   help="output root (default: folder containing this script)")
    p.add_argument("--num-images", type=int, default=1000)
    p.add_argument("--size", type=int, default=256, help="image side in pixels")
    p.add_argument("--n-views", type=int, default=360,
                   help="number of projection angles")
    p.add_argument("--scan-range", type=float, default=360.0,
                   help="angular range in degrees (360 -> full rings)")
    p.add_argument("--max-attenuation", type=float, default=4.0,
                   help="line integral across full FOV diameter at mu=1")
    p.add_argument("--strength", type=float, default=1.0,
                   help="global multiplier on detector-error magnitudes")
    p.add_argument("--sl-fraction", type=float, default=0.4,
                   help="fraction of Shepp–Logan phantoms (rest: tissue phantoms)")
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--format", choices=["npy", "png"], default="npy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--overwrite", action="store_true",
                   help="delete existing training/ and testing/ first")
    args = p.parse_args()

    root = os.path.abspath(args.out)
    for split in ("training", "testing"):
        for kind in ("noisy", "clean"):
            d = os.path.join(root, split, kind)
            if args.overwrite and os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    cfg = dict(size=args.size, n_views=args.n_views, strength=args.strength,
               scan_range=args.scan_range,
               max_attenuation=args.max_attenuation, sl_fraction=args.sl_fraction)
    seeds = np.random.SeedSequence(args.seed).generate_state(args.num_images)
    jobs = [(i, int(seeds[i]), cfg) for i in range(args.num_images)]

    perm = np.random.default_rng(args.seed + 1).permutation(args.num_images)
    n_train = int(round(args.train_fraction * args.num_images))
    split_of = {int(i): ("training" if r < n_train else "testing")
                for r, i in enumerate(perm)}

    # Fixed global intensity window (phantom units). Phantom values are in
    # [0, ~1.3]; FBP undershoot and rings go slightly negative. A fixed window
    # (not per-image min/max) keeps noisy and clean on the same scale and
    # matches what is available at inference time.
    lo, hi = -0.2, 1.4

    t0 = time.time()
    rows = []
    clip_frac = []
    with Pool(args.workers) as pool:
        for k, (idx, kind, clean, noisy, prm) in enumerate(
                pool.imap_unordered(_work, jobs, chunksize=4)):
            split = split_of[idx]
            name = f"{idx:05d}"
            for tag, img in (("clean", clean), ("noisy", noisy)):
                base = os.path.join(root, split, tag, name)
                if args.format == "npy":
                    np.save(base + ".npy", img)
                else:
                    to_png16(img, lo, hi, base + ".png")
            clip_frac.append(float(np.mean((noisy < lo) | (noisy > hi))))
            ring = noisy - clean
            rows.append(dict(name=name, split=split, phantom=kind,
                             ring_rms=float(np.sqrt(np.mean(ring ** 2))),
                             ring_max=float(np.abs(ring).max()), **asdict(prm)))
            if (k + 1) % 50 == 0 or k + 1 == args.num_images:
                print(f"  {k + 1}/{args.num_images}  ({time.time() - t0:.1f}s)")

    rows.sort(key=lambda r: r["name"])
    with open(os.path.join(root, "ring_params.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = dict(format=args.format, size=args.size, n_views=args.n_views,
                scan_range=args.scan_range,
                max_attenuation=args.max_attenuation, strength=args.strength,
                window=[lo, hi], png_is_normalized=(args.format == "png"),
                num_train=n_train, num_test=args.num_images - n_train,
                seed=args.seed,
                mean_ring_rms=float(np.mean([r["ring_rms"] for r in rows])),
                mean_fraction_clipped_by_window=float(np.mean(clip_frac)))
    with open(os.path.join(root, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done: {n_train} train / {args.num_images - n_train} test "
          f"in {time.time() - t0:.1f}s -> {root}")
    print(f"Mean ring RMS (phantom units): {meta['mean_ring_rms']:.4f}; "
          f"pixels outside window: {100 * meta['mean_fraction_clipped_by_window']:.3f}%")


if __name__ == "__main__":
    main()
