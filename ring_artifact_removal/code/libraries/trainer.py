"""
trainer.py — training / validation loop, checkpointing, logging, evaluation.

* Optimiser: Adam (betas 0.9, 0.999), optional weight decay.
* LR schedule: 'cosine' (CosineAnnealingLR with linear warm-up) or
  'plateau' (ReduceLROnPlateau on validation loss).
* Mixed precision (bf16/fp16 autocast) on CUDA; gradient clipping.
* Checkpoints: last.pt every epoch, best.pt on best validation PSNR.
  Resuming restores model, optimiser, scheduler, scaler and history.
* Logs: training_log.csv (one row per epoch) + training_curves.png.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import AverageMeter, compute_metrics, save_comparison, save_training_curves


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module,
                 train_loader: DataLoader, val_loader: Optional[DataLoader],
                 device: torch.device, out_dir: str, lr: float = 5e-4,
                 weight_decay: float = 0.0, epochs: int = 100,
                 scheduler: str = "cosine", warmup_epochs: int = 2,
                 grad_clip: float = 1.0, amp: bool = True,
                 log_every: int = 20, early_stop_patience: int = 0):
        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.out_dir = out_dir
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.early_stop_patience = early_stop_patience
        os.makedirs(out_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          betas=(0.9, 0.999), weight_decay=weight_decay)
        self.scheduler_name = scheduler
        if scheduler == "cosine":
            steps_per_epoch = max(len(train_loader), 1)
            warm = warmup_epochs * steps_per_epoch
            total = epochs * steps_per_epoch

            def lr_lambda(step):
                if step < warm:
                    return (step + 1) / warm
                prog = (step - warm) / max(total - warm, 1)
                return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            self.step_per_batch = True
        elif scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
            self.step_per_batch = False
        else:
            raise ValueError(f"Unknown scheduler '{scheduler}'")

        self.use_amp = amp and device.type == "cuda"
        self.amp_dtype = (torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported()
                          else torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and
                                           self.amp_dtype == torch.float16)

        self.start_epoch = 1
        self.best_psnr = -float("inf")
        self.best_epoch = 0
        self.history = []

    # ------------------------------------------------------------------ io
    def save_checkpoint(self, path: str, epoch: int):
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_psnr": self.best_psnr,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }, path)

    def load_checkpoint(self, path: str, weights_only: bool = False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if not weights_only:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.start_epoch = ckpt["epoch"] + 1
            self.best_psnr = ckpt.get("best_psnr", -float("inf"))
            self.best_epoch = ckpt.get("best_epoch", 0)
            self.history = ckpt.get("history", [])
        print(f"[trainer] loaded {path} (epoch {ckpt['epoch']})")
        return ckpt

    # --------------------------------------------------------------- loops
    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                              enabled=self.use_amp)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meters = {k: AverageMeter() for k in ("loss", "l1", "ssim_loss")}
        t0 = time.time()
        n_batches = len(self.train_loader)
        for it, (noisy, clean, _) in enumerate(self.train_loader, 1):
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                pred = self.model(noisy)
            # loss in fp32: SSIM's variance terms are cancellation-prone in half
            loss, parts = self.criterion(pred.float(), clean)

            if not torch.isfinite(loss):
                print(f"[trainer] non-finite loss at epoch {epoch} iter {it}; batch skipped")
                continue
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.step_per_batch:
                self.scheduler.step()

            bs = noisy.shape[0]
            meters["loss"].update(loss.item(), bs)
            for k in ("l1", "ssim_loss"):
                meters[k].update(parts[k].item(), bs)
            if self.log_every and (it % self.log_every == 0 or it == n_batches):
                print(f"  ep {epoch:3d} [{it:4d}/{n_batches}] loss {meters['loss'].avg:.5f} "
                      f"l1 {meters['l1'].avg:.5f} lr {self.optimizer.param_groups[0]['lr']:.2e} "
                      f"({time.time() - t0:.0f}s)")
        return {k: m.avg for k, m in meters.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, return_per_image: bool = False):
        """Mean loss and metrics for restored AND noisy-input baseline."""
        self.model.eval()
        loss_m = AverageMeter()
        acc = {f"{w}_{k}": [] for w in ("restored", "noisy")
               for k in ("psnr", "ssim", "mae", "rre")}
        names = []
        for noisy, clean, name in loader:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            with self._autocast():
                pred = self.model(noisy)
            pred = pred.float()
            loss, _ = self.criterion(pred, clean)
            loss_m.update(loss.item(), noisy.shape[0])
            for w, x in (("restored", pred), ("noisy", noisy)):
                for k, v in compute_metrics(x, clean).items():
                    acc[f"{w}_{k}"].append(v.cpu())
            names += list(name)
        acc = {k: torch.cat(v).numpy() for k, v in acc.items()}
        summary = {"loss": loss_m.avg, **{k: float(np.mean(v)) for k, v in acc.items()}}
        if return_per_image:
            return summary, acc, names
        return summary

    # ------------------------------------------------------------------ fit
    def fit(self):
        log_path = os.path.join(self.out_dir, "training_log.csv")
        print(f"[trainer] device {self.device} | amp {self.use_amp} "
              f"({self.amp_dtype if self.use_amp else 'fp32'}) | epochs "
              f"{self.start_epoch}..{self.epochs}")
        epochs_no_improve = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            tr = self.train_one_epoch(epoch)
            row = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"],
                   "train_loss": tr["loss"], "train_l1": tr["l1"]}

            if self.val_loader is not None:
                va = self.evaluate(self.val_loader)
                row.update(val_loss=va["loss"], val_psnr=va["restored_psnr"],
                           val_ssim=va["restored_ssim"], val_rre=va["restored_rre"],
                           val_noisy_psnr=va["noisy_psnr"])
                score = va["restored_psnr"]
                if not self.step_per_batch:
                    self.scheduler.step(va["loss"])
            else:
                score = -tr["loss"]
                if not self.step_per_batch:
                    self.scheduler.step(tr["loss"])
            row["time_s"] = time.time() - t0
            self.history.append(row)

            improved = score > self.best_psnr
            if improved:
                self.best_psnr, self.best_epoch = score, epoch
                self.save_checkpoint(os.path.join(self.out_dir, "best.pt"), epoch)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            self.save_checkpoint(os.path.join(self.out_dir, "last.pt"), epoch)

            msg = f"[epoch {epoch:3d}] train {tr['loss']:.5f}"
            if self.val_loader is not None:
                msg += (f" | val {row['val_loss']:.5f} PSNR {row['val_psnr']:.2f} dB "
                        f"(input {row['val_noisy_psnr']:.2f}) SSIM {row['val_ssim']:.4f} "
                        f"RRE {row['val_rre']:.5f}")
            msg += f" | {row['time_s']:.0f}s" + ("  *best*" if improved else "")
            print(msg)

            write_header = not os.path.isfile(log_path) or epoch == 1
            with open(log_path, "w" if epoch == 1 else "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)
            save_training_curves(self.history, os.path.join(self.out_dir, "training_curves.png"))

            if self.early_stop_patience and epochs_no_improve >= self.early_stop_patience:
                print(f"[trainer] early stop: no improvement for {epochs_no_improve} epochs")
                break

        print(f"[trainer] best epoch {self.best_epoch} (score {self.best_psnr:.3f})")
        return self.history

    # ------------------------------------------------------------- testing
    @torch.no_grad()
    def test(self, loader: DataLoader, out_dir: str, n_plots: int = 10,
             save_outputs: bool = False):
        """Full test-set evaluation + per-image CSV + comparison figures."""
        os.makedirs(out_dir, exist_ok=True)
        summary, per_img, names = self.evaluate(loader, return_per_image=True)

        with open(os.path.join(out_dir, "test_metrics_per_image.csv"), "w", newline="") as f:
            keys = list(per_img.keys())
            w = csv.writer(f)
            w.writerow(["name"] + keys)
            for i, n in enumerate(names):
                w.writerow([n] + [f"{per_img[k][i]:.6f}" for k in keys])

        # paired statistics: improvement per image, with a 95% bootstrap CI
        stats = {}
        rng = np.random.default_rng(0)
        for k in ("psnr", "ssim", "mae", "rre"):
            d = per_img[f"restored_{k}"] - per_img[f"noisy_{k}"]
            boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            stats[f"delta_{k}"] = {"mean": float(d.mean()), "std": float(d.std(ddof=1))
                                   if len(d) > 1 else 0.0,
                                   "ci95": [float(np.percentile(boots, 2.5)),
                                            float(np.percentile(boots, 97.5))],
                                   "frac_improved": float(np.mean(d > 0) if k in ("psnr", "ssim")
                                                          else np.mean(d < 0))}
        result = {"n_images": len(names), "summary": summary, "paired_improvement": stats}
        with open(os.path.join(out_dir, "test_summary.json"), "w") as f:
            json.dump(result, f, indent=2)

        # figures (+ optional raw outputs)
        plot_dir = os.path.join(out_dir, "comparisons")
        npy_dir = os.path.join(out_dir, "restored_npy")
        if save_outputs:
            os.makedirs(npy_dir, exist_ok=True)
        self.model.eval()
        done = 0
        for noisy, clean, name in loader:
            if done >= n_plots and not save_outputs:
                break
            with self._autocast():
                pred = self.model(noisy.to(self.device))
            pred = pred.float().clamp(0, 1).cpu()
            for b in range(noisy.shape[0]):
                if save_outputs:
                    np.save(os.path.join(npy_dir, f"{name[b]}.npy"), pred[b, 0].numpy())
                if done < n_plots:
                    i = names.index(name[b])
                    m_n = {"psnr": per_img["noisy_psnr"][i], "ssim": per_img["noisy_ssim"][i]}
                    m_r = {"psnr": per_img["restored_psnr"][i],
                           "ssim": per_img["restored_ssim"][i]}
                    save_comparison(clean[b, 0].numpy(), noisy[b, 0].numpy(),
                                    pred[b, 0].numpy(),
                                    os.path.join(plot_dir, f"{name[b]}.png"),
                                    title=f"Test sample {name[b]}",
                                    metrics_noisy=m_n, metrics_restored=m_r)
                    done += 1
        return result
