# CT Ring Artifact Removal

Residual U-Net that removes ring artifacts from reconstructed 2D CT slices,
trained on synthetic paired data simulated in the sinogram domain.

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate data (1000 slices, 256x256, 80/20 train/test)
python data/generate_data.py --num-images 1000 --size 256 --workers 8

# 2. Train + evaluate on the held-out test set
python code/main.py --epochs 60 --batch-size 8 --lr 5e-4 --save-dir runs/exp

# Evaluate a saved model only
python code/main.py --eval-only --checkpoint runs/exp/best.pt --save-dir runs/exp
```

On CPU use `--model unet_small` and a smaller dataset (e.g. `--size 128`).

## How the rings are simulated

For detector channel i and view angle θ (see `data/generate_data.py`):

```
p(i,θ)   = ∫ μ dl                         ideal line integral
Im(i,θ)  = g_i(θ) · I0 e^{-p} + o_i        gain error g_i = 1 + ε_i, offset o_i
pm(i,θ)  = −ln(Im / I0)
         ≈ p − ε_i  (+ o_i e^{p} / I0)     constant in θ → vertical stripe
FBP(stripe at offset s) → circle of radius |s| about the rotation axis
```

`ε_i` = small Gaussian error on every channel + a few strong defective
channel groups (1–5 channels wide, 0.8–4 %); `o_i` gives rings that are
stronger behind dense material; 30 % of scans add a linear gain drift.
Clean target = FBP of the uncorrupted sinogram with identical geometry, so
the network only has to remove rings, not FBP blur.

## Normalisation

A single fixed window `[lo, hi] = [-0.2, 1.4]` (phantom units, stored in
`data/dataset_meta.json`) maps both noisy and clean images to [0, 1]. Per-image
min-max scaling is avoided because rings shift the extrema of the noisy image.

## Model

Residual U-Net (`code/libraries/models.py`): `restored = noisy − s·f(noisy)`,
pre-activation residual blocks with GroupNorm, depth 4, learned strided
down-sampling, bilinear up-sampling + skip concatenation, and two
coordinate channels (radius, constant) so the network knows where the ring
centre is. `s = 0.02` keeps the network output in O(1) units — without it
Adam's step size is larger than the ~1e-3 artifact amplitudes and the model
stays at the identity.

## Metrics

PSNR, SSIM (Gaussian σ = 1.5), MAE, and a ring-specific **ring residual
error** (RMS over radius of the angular mean of the error). Test reports
include paired per-image improvements with 95 % bootstrap CIs.
Validation (10 % of `training/`) drives checkpoint selection; `testing/` is
used once at the end.

## Known limitations

* Parallel-beam geometry with ring centre exactly at the image centre. Real
  scans need the centre of rotation, and fan-beam rebinning if the
  corruption is applied to fan-beam data.
* No quantum noise or other artifacts; a model trained here should be
  tested on noisy data before trusting it on real scans.
* Synthetic ellipse phantoms lack real anatomical texture. Fine-tune on real
  paired data (e.g. flat-field-corrected vs. raw) where available.
