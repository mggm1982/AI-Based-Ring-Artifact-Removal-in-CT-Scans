"""Ring artifact removal library: data loading, models, losses/metrics, training."""

from .dataset import RingArtifactDataset, get_dataloaders, load_meta
from .models import RingRemovalUNet, build_model, count_parameters
from .trainer import Trainer
from .utils import (L1SSIMLoss, compute_metrics, get_device, psnr,
                    ring_residual_error, save_comparison, set_seed, ssim)

__all__ = [
    "RingArtifactDataset", "get_dataloaders", "load_meta",
    "RingRemovalUNet", "build_model", "count_parameters",
    "Trainer",
    "L1SSIMLoss", "compute_metrics", "get_device", "psnr", "ring_residual_error",
    "save_comparison", "set_seed", "ssim",
]
