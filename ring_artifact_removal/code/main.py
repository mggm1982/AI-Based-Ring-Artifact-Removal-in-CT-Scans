#!/usr/bin/env python3
"""
main.py — train and evaluate a ring-artifact-removal network.

Typical use (from the project root):

    python data/generate_data.py --num-images 1000
    python code/main.py --epochs 60 --batch-size 8 --lr 5e-4

Evaluate an existing checkpoint only:

    python code/main.py --eval-only --checkpoint runs/exp/best.pt

Outputs in --save-dir:
    best.pt, last.pt              checkpoints (best = highest val PSNR)
    config.json                   all CLI arguments
    training_log.csv              per-epoch metrics
    training_curves.png
    test/test_summary.json        test-set means + paired improvement w/ 95% CI
    test/test_metrics_per_image.csv
    test/comparisons/*.png        [GT | noisy | restored | |restored - GT|]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from libraries import (L1SSIMLoss, Trainer, build_model, count_parameters,  # noqa: E402
                       get_dataloaders, get_device, set_seed)


def parse_args():
    root = os.path.dirname(HERE)
    p = argparse.ArgumentParser(description="CT ring artifact removal (residual U-Net)")
    g = p.add_argument_group("data")
    g.add_argument("--data-path", default=os.path.join(root, "data"),
                   help="folder containing training/ and testing/")
    g.add_argument("--val-fraction", type=float, default=0.1,
                   help="fraction of training/ held out for validation")
    g.add_argument("--no-augment", action="store_true")
    g.add_argument("--num-workers", type=int, default=2)

    g = p.add_argument_group("model")
    g.add_argument("--model", choices=["unet", "unet_small"], default="unet")
    g.add_argument("--base-channels", type=int, default=32)
    g.add_argument("--depth", type=int, default=4)
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--no-coords", action="store_true",
                   help="disable radius coordinate input channels")
    g.add_argument("--residual-scale", type=float, default=0.02,
                   help="scale on the predicted artifact (see models.py)")

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=60)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine")
    g.add_argument("--warmup-epochs", type=int, default=2)
    g.add_argument("--ssim-weight", type=float, default=0.84,
                   help="alpha in (1-alpha)*L1 + alpha*(1-SSIM)")
    g.add_argument("--ring-weight", type=float, default=0.0,
                   help="weight of the angular-mean ring-profile loss term")
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--no-amp", action="store_true")
    g.add_argument("--early-stop", type=int, default=0,
                   help="stop after N epochs without val improvement (0 = off)")

    g = p.add_argument_group("run")
    g.add_argument("--save-dir", "--model-save-path", dest="save_dir",
                   default=os.path.join(root, "runs", "exp"))
    g.add_argument("--resume", default=None, help="checkpoint to resume training from")
    g.add_argument("--eval-only", action="store_true")
    g.add_argument("--checkpoint", default=None,
                   help="weights for --eval-only (default: <save-dir>/best.pt)")
    g.add_argument("--n-plots", type=int, default=12, help="test comparison figures")
    g.add_argument("--save-outputs", action="store_true",
                   help="also save every restored test image as .npy")
    g.add_argument("--device", default="auto")
    g.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        args.data_path, batch_size=args.batch_size, val_fraction=args.val_fraction,
        num_workers=args.num_workers, seed=args.seed, augment=not args.no_augment)

    model = build_model(args.model, base=args.base_channels, depth=args.depth,
                        dropout=args.dropout, coord_channels=not args.no_coords,
                        residual_scale=args.residual_scale)
    print(f"[main] model {args.model}: {count_parameters(model) / 1e6:.2f} M parameters")

    criterion = L1SSIMLoss(alpha=args.ssim_weight, ring_weight=args.ring_weight)
    trainer = Trainer(model, criterion, train_loader, val_loader, device, args.save_dir,
                      lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs,
                      scheduler=args.scheduler, warmup_epochs=args.warmup_epochs,
                      grad_clip=args.grad_clip, amp=not args.no_amp,
                      early_stop_patience=args.early_stop)

    if args.eval_only:
        ckpt = args.checkpoint or os.path.join(args.save_dir, "best.pt")
        trainer.load_checkpoint(ckpt, weights_only=True)
    else:
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.fit()
        best = os.path.join(args.save_dir, "best.pt")
        if os.path.isfile(best):
            trainer.load_checkpoint(best, weights_only=True)

    print("[main] evaluating on held-out test set ...")
    res = trainer.test(test_loader, os.path.join(args.save_dir, "test"),
                       n_plots=args.n_plots, save_outputs=args.save_outputs)
    s, d = res["summary"], res["paired_improvement"]
    print(f"\n=== Test set ({res['n_images']} images) ===")
    print(f"{'metric':<8}{'noisy input':>14}{'restored':>12}{'Δ mean [95% CI]':>32}")
    for k, unit in (("psnr", "dB"), ("ssim", ""), ("mae", ""), ("rre", "")):
        ci = d[f"delta_{k}"]["ci95"]
        print(f"{k.upper():<8}{s['noisy_' + k]:>14.4f}{s['restored_' + k]:>12.4f}"
              f"{d['delta_' + k]['mean']:>+14.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] {unit}")
    print(f"\nResults written to {args.save_dir}")


if __name__ == "__main__":
    main()
