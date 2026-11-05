"""
dataset.py — paired (noisy, clean) CT slice loading.

Normalisation
-------------
Both images of a pair are mapped to [0, 1] with the SAME fixed intensity
window [lo, hi] read from data/dataset_meta.json (written by
generate_data.py):

        x_norm = clip( (x - lo) / (hi - lo), 0, 1 )

Per-image min/max scaling is deliberately avoided: the ring component shifts
the noisy image's extrema, so per-image scaling would put noisy and clean on
different intensity scales and the network would have to learn a global
gain/offset in addition to the rings. A fixed window is also what is
available at inference time, when no clean reference exists.

16-bit PNGs written by generate_data.py --format png are already normalised;
they are divided by 65535. 8-bit PNGs are divided by 255.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = (".npy", ".png", ".tif", ".tiff")


def load_meta(data_root: str) -> dict:
    """Reads dataset_meta.json; falls back to a default window if absent."""
    path = os.path.join(data_root, "dataset_meta.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"window": [-0.2, 1.4], "png_is_normalized": True}


def _read_image(path: str) -> Tuple[np.ndarray, bool]:
    """Returns (float32 array, already_normalised)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32), False
    from PIL import Image
    arr = np.array(Image.open(path))
    if arr.ndim == 3:                           # RGB(A) -> luminance
        arr = arr[..., :3].mean(axis=-1)
    if arr.dtype == np.uint16 or arr.max() > 255:
        return arr.astype(np.float32) / 65535.0, True
    return arr.astype(np.float32) / 255.0, True


class RingArtifactDataset(Dataset):
    """Paired dataset reading <split_dir>/noisy/* and <split_dir>/clean/*.

    Files are paired by basename (without extension). Each item is
    (noisy, clean, name) with tensors of shape (1, H, W) in [0, 1].

    Augmentation (training only): random horizontal/vertical flips and
    90-degree rotations. These are symmetries of the ring geometry: rings are
    centred on the image centre (the rotation axis), so flips and rot90 keep
    them centred and circular. Random crops/translations are NOT used because
    they would move the ring centre, which is a useful prior for the network.
    """

    def __init__(self, split_dir: str, window: Sequence[float] = (-0.2, 1.4),
                 augment: bool = False, names: Optional[List[str]] = None):
        self.noisy_dir = os.path.join(split_dir, "noisy")
        self.clean_dir = os.path.join(split_dir, "clean")
        if not (os.path.isdir(self.noisy_dir) and os.path.isdir(self.clean_dir)):
            raise FileNotFoundError(f"Expected noisy/ and clean/ under {split_dir}")

        def index(d):
            return {os.path.splitext(f)[0]: os.path.join(d, f)
                    for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)}

        noisy, clean = index(self.noisy_dir), index(self.clean_dir)
        common = sorted(set(noisy) & set(clean))
        if names is not None:
            common = [n for n in common if n in set(names)]
        if not common:
            raise RuntimeError(f"No paired images found in {split_dir}")
        missing = (set(noisy) ^ set(clean))
        if missing:
            print(f"[dataset] warning: {len(missing)} unpaired files ignored "
                  f"in {split_dir}")

        self.names = common
        self.noisy_paths = [noisy[n] for n in common]
        self.clean_paths = [clean[n] for n in common]
        self.lo, self.hi = float(window[0]), float(window[1])
        self.augment = augment

    def __len__(self) -> int:
        return len(self.names)

    def _normalise(self, arr: np.ndarray, already: bool) -> np.ndarray:
        if not already:
            arr = (arr - self.lo) / (self.hi - self.lo)
        return np.clip(arr, 0.0, 1.0)

    def denormalise(self, x: np.ndarray) -> np.ndarray:
        """Back to phantom (attenuation) units, for .npy-based data."""
        return x * (self.hi - self.lo) + self.lo

    def __getitem__(self, i: int):
        n, nn = _read_image(self.noisy_paths[i])
        c, cn = _read_image(self.clean_paths[i])
        if n.shape != c.shape:
            raise ValueError(f"Shape mismatch for {self.names[i]}: {n.shape} vs {c.shape}")
        n, c = self._normalise(n, nn), self._normalise(c, cn)

        if self.augment:
            if np.random.rand() < 0.5:
                n, c = n[:, ::-1], c[:, ::-1]
            if np.random.rand() < 0.5:
                n, c = n[::-1, :], c[::-1, :]
            k = np.random.randint(4)
            if k:
                n, c = np.rot90(n, k), np.rot90(c, k)

        n = torch.from_numpy(np.ascontiguousarray(n))[None]
        c = torch.from_numpy(np.ascontiguousarray(c))[None]
        return n, c, self.names[i]


def get_dataloaders(data_root: str, batch_size: int = 8, val_fraction: float = 0.1,
                    num_workers: int = 2, seed: int = 0, augment: bool = True):
    """Builds train / validation / test loaders.

    The validation set is carved out of data/training (by a seeded random
    split) and is used for checkpoint selection and LR scheduling. The
    data/testing set is touched only once, for final evaluation, so the
    reported test metrics are not biased by model selection.
    """
    meta = load_meta(data_root)
    window = meta.get("window", [-0.2, 1.4])
    train_dir = os.path.join(data_root, "training")
    test_dir = os.path.join(data_root, "testing")

    full = RingArtifactDataset(train_dir, window, augment=False)
    n_val = max(1, int(round(val_fraction * len(full)))) if val_fraction > 0 else 0
    perm = np.random.default_rng(seed).permutation(len(full))
    val_names = [full.names[i] for i in perm[:n_val]]
    train_names = [full.names[i] for i in perm[n_val:]]

    train_ds = RingArtifactDataset(train_dir, window, augment=augment, names=train_names)
    val_ds = RingArtifactDataset(train_dir, window, augment=False, names=val_names) \
        if n_val else None
    test_ds = RingArtifactDataset(test_dir, window, augment=False)

    pin = torch.cuda.is_available()
    common = dict(num_workers=num_workers, pin_memory=pin,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=len(train_ds) > batch_size, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common) \
        if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    print(f"[dataset] train {len(train_ds)} | val {len(val_ds) if val_ds else 0} | "
          f"test {len(test_ds)} | window {window}")
    return train_loader, val_loader, test_loader, meta
