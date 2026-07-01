"""
train_sr.py
─────────────────────────────────────────────────────────────────────────────
Phase 2: Super-Resolution Training Script

Trains an RRDBNet to upscale TIR@200m (256×256) → TIR@100m (512×512).

Usage:
    # Full training (recommended on GPU):
    python train_sr.py --patches_dir output/patches --epochs 50

    # Quick sanity-check (2 epochs, batch=2):
    python train_sr.py --epochs 2 --batch_size 2 --num_workers 0

    # Resume from a checkpoint:
    python train_sr.py --resume checkpoints/best_rrdbnet.pth

    # CPU-only:
    python train_sr.py --epochs 10 --device cpu

Outputs:
    checkpoints/best_rrdbnet.pth   ← best validation-loss checkpoint
    checkpoints/last_rrdbnet.pth   ← latest epoch checkpoint
    output/sr_results/train_log.csv ← per-epoch loss / PSNR / SSIM
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import csv
import math
import time
import logging
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset import build_dataloaders
from src.models.rrdbnet import build_model

# ── Logging setup ─────────────────────────────────────────────────────────
os.makedirs("output/sr_results", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/sr_results/train.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.
    Higher is better. PSNR > 30 dB is generally considered good for SR.

    Formula: PSNR = 10 * log10(MAX² / MSE)
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0  # perfect reconstruction
    return 10.0 * math.log10((max_val ** 2) / mse)


def ssim_fast(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Simplified SSIM (Structural Similarity Index) for 1-channel images.
    Uses global statistics (fast but approximate).

    SSIM = 1.0 means perfect structural match.
    SSIM > 0.85 is a strong result for TIR SR.

    Full SSIM requires a sliding window — this version is fast enough for
    per-epoch logging without adding heavy dependencies.
    """
    C1, C2 = (0.01 ** 2), (0.03 ** 2)
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x  = pred.var()
    sigma_y  = target.var()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()

    numerator   = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return (numerator / (denominator + 1e-8)).item()


# ── Validation loop ───────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: nn.Module, val_loader, device: str, criterion: nn.Module):
    """Run one full pass over the validation set. Returns (avg_loss, avg_psnr, avg_ssim)."""
    model.eval()
    total_loss = total_psnr = total_ssim = 0.0

    for lr, hr in val_loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        sr = model(lr)
        sr = torch.clamp(sr, 0.0, 1.0)

        loss = criterion(sr, hr).item()
        total_loss += loss
        total_psnr += psnr(sr, hr)
        total_ssim += ssim_fast(sr, hr)

    n = max(len(val_loader), 1)
    return total_loss / n, total_psnr / n, total_ssim / n


# ── Training loop ─────────────────────────────────────────────────────────

def train(args):
    # ── Device ──
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──
    logger.info("Building dataloaders …")
    train_loader, val_loader = build_dataloaders(
        patches_root=args.patches_dir,
        val_scenes=args.val_scenes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if len(train_loader.dataset) == 0:
        logger.error("No training samples found. Check output/patches/ structure.")
        sys.exit(1)

    # ── Model ──
    model = build_model(
        num_block=args.num_block,
        num_feat=args.num_feat,
        scale=2,
        device=device,
    )

    # ── Loss: Charbonnier (smooth L1) — more robust to outliers than L1/L2 ──
    # Charbonnier loss: sqrt((x-y)² + ε²), reduces to L1 for large errors,
    # L2 near zero. It handles nodata / cloud pixels gracefully.
    class CharbonnierLoss(nn.Module):
        def __init__(self, eps=1e-6):
            super().__init__()
            self.eps = eps
        def forward(self, pred, target):
            return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))

    criterion = CharbonnierLoss().to(device)

    # ── Optimiser: Adam with weight decay ──
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )

    # ── Scheduler: Cosine Annealing (LR decays from lr → lr/1000 over epochs) ──
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    # ── Resume from checkpoint ──
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed at epoch {start_epoch}, best val_loss = {best_val_loss:.6f}")

    # ── CSV log ──
    os.makedirs("output/sr_results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    csv_path = "output/sr_results/train_log.csv"
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if start_epoch == 1:
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr_db", "val_ssim", "lr", "time_s"])

    logger.info(
        f"\n{'='*60}\n"
        f"  Phase 2 SR Training — RRDBNet x2\n"
        f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}\n"
        f"  Train samples: {len(train_loader.dataset)}  |  Val samples: {len(val_loader.dataset)}\n"
        f"{'='*60}"
    )

    # ── Main training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()

            # Gradient clipping prevents exploding gradients in dense blocks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                logger.info(
                    f"  Epoch {epoch}/{args.epochs} "
                    f"[{batch_idx+1}/{len(train_loader)}] "
                    f"loss={loss.item():.6f}"
                )

        scheduler.step()
        avg_train_loss = epoch_loss / max(len(train_loader), 1)

        # ── Validation ──
        avg_val_loss, avg_val_psnr, avg_val_ssim = validate(
            model, val_loader, device, criterion
        )

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={avg_train_loss:.6f} | "
            f"val_loss={avg_val_loss:.6f} | "
            f"PSNR={avg_val_psnr:.2f} dB | "
            f"SSIM={avg_val_ssim:.4f} | "
            f"LR={current_lr:.2e} | "
            f"time={elapsed:.1f}s"
        )

        csv_writer.writerow([
            epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}",
            f"{avg_val_psnr:.4f}", f"{avg_val_ssim:.4f}",
            f"{current_lr:.2e}", f"{elapsed:.1f}",
        ])
        csv_file.flush()

        # ── Save checkpoints ──
        ckpt = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "args":          vars(args),
        }

        # Always save latest
        torch.save(ckpt, "checkpoints/last_rrdbnet.pth")

        # Save best (lowest validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, "checkpoints/best_rrdbnet.pth")
            logger.info(f"  ✓ New best! val_loss={best_val_loss:.6f} — checkpoint saved.")

    csv_file.close()
    logger.info(
        f"\nTraining complete.\n"
        f"Best val_loss = {best_val_loss:.6f}\n"
        f"Checkpoints: checkpoints/best_rrdbnet.pth\n"
        f"Log:         output/sr_results/train_log.csv\n"
    )


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: Train RRDBNet for TIR Super-Resolution (×2)"
    )
    parser.add_argument(
        "--patches_dir", type=str, default="output/patches",
        help="Path to the output/patches directory (default: output/patches)"
    )
    parser.add_argument(
        "--val_scenes", type=str, nargs="+",
        default=["SCENE_009", "SCENE_010"],
        help="Scene IDs to use for validation (default: SCENE_009 SCENE_010)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Total training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for training (default: 4; reduce to 2 on CPU)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Initial learning rate for Adam (default: 2e-4)"
    )
    parser.add_argument(
        "--num_block", type=int, default=6,
        help="Number of RRDB blocks (default: 6; use 23 for full Real-ESRGAN)"
    )
    parser.add_argument(
        "--num_feat", type=int, default=64,
        help="Feature channels in RRDBNet (default: 64)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="DataLoader worker processes (default: 2; set 0 for debugging)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to train on. 'auto' picks GPU if available (default: auto)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint .pth file to resume training from"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("output/sr_results", exist_ok=True)
    train(args)
