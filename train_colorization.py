"""
train_colorization.py  --  Phase 3 + 4: Pix2Pix cGAN Training
TIR@100m (super-resolved) -> RGB@100m colorization

Loss = lambda_L1 * L1  +  lambda_GAN * adversarial  +  lambda_physics * physics
     (lambda_L1=100 keeps GAN stable on small datasets)
     (lambda_physics=5 enforces Stefan-Boltzmann thermal color constraints)

Usage:
    python train_colorization.py                             # full training
    python train_colorization.py --epochs 200 --batch_size 1
    python train_colorization.py --resume checkpoints/last_pix2pix.pth
    python train_colorization.py --epochs 5 --num_workers 0  # CPU quick test
    python train_colorization.py --lambda_physics 5.0         # with physics loss
"""

import os
import sys
import csv
import time
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.pix2pix import UNetGenerator, PatchGANDiscriminator
from src.colorization_dataset import build_color_dataloaders
from src.physics_loss import PhysicsInformedLoss

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3: Pix2Pix Colorization Training")
    p.add_argument("--patches_dir", default="output/patches")
    p.add_argument("--val_scenes",  nargs="+", default=["SCENE_009", "SCENE_010"])
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=1)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--lambda_l1",   type=float, default=100.0,
                   help="Weight for L1 pixel loss (default 100, standard Pix2Pix)")
    p.add_argument("--lambda_gan",  type=float, default=1.0,
                   help="Weight for adversarial loss")
    p.add_argument("--lambda_physics", type=float, default=5.0,
                   help="Weight for physics-informed loss (0=disabled)")
    p.add_argument("--ngf",         type=int,   default=64, help="Generator base filters")
    p.add_argument("--ndf",         type=int,   default=64, help="Discriminator base filters")
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--results_dir",    default="output/colorization_results")
    p.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")
    p.add_argument("--device",      default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--decay_after", type=int,   default=100,
                   help="Epoch from which LR linearly decays to 0")
    return p.parse_args()


# ─── GAN Losses ───────────────────────────────────────────────────────────────

def discriminator_loss(D, tir_cond, rgb_real, rgb_fake, criterion):
    """Hinge-style BCE loss with one-sided label smoothing."""
    pred_real = D(tir_cond, rgb_real)
    pred_fake = D(tir_cond, rgb_fake.detach())
    # Label smoothing: real=0.9, fake=0.0
    real_labels = torch.full_like(pred_real, 0.9)
    fake_labels = torch.zeros_like(pred_fake)
    loss_real = criterion(pred_real, real_labels)
    loss_fake = criterion(pred_fake, fake_labels)
    return (loss_real + loss_fake) * 0.5


def generator_loss(D, tir_cond, rgb_fake, rgb_real, criterion_gan, criterion_l1,
                   lambda_l1, lambda_gan, physics_fn=None, lambda_physics=0.0):
    """G tries to fool D, minimise L1, and satisfy physics constraints."""
    pred_fake   = D(tir_cond, rgb_fake)
    loss_gan    = criterion_gan(pred_fake, torch.ones_like(pred_fake))
    loss_l1     = criterion_l1(rgb_fake, rgb_real)
    loss_phys   = torch.tensor(0.0, device=rgb_fake.device)
    if physics_fn is not None and lambda_physics > 0:
        loss_phys = physics_fn(tir_cond, rgb_fake)   # tir_cond is [0,1] normalized
    total = lambda_gan * loss_gan + lambda_l1 * loss_l1 + lambda_physics * loss_phys
    return total, loss_gan, loss_l1, loss_phys


# ─── SSIM (simple 2D, single-image) ──────────────────────────────────────────

def ssim_batch(pred, target):
    """Batch-averaged SSIM on [-1,1] tensors (rough estimate, no window)."""
    pred   = (pred   + 1) / 2   # -> [0,1]
    target = (target + 1) / 2
    mu1, mu2 = pred.mean(), target.mean()
    s1,  s2  = pred.std(),  target.std()
    cov  = ((pred - mu1) * (target - mu2)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return ((2*mu1*mu2 + c1) * (2*cov + c2) /
            ((mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2))).item()


# ─── LR Scheduler (linear decay after decay_after epochs) ────────────────────

def make_lr_lambda(total_epochs, decay_after):
    def lr_lambda(epoch):
        if epoch < decay_after:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_after) / (total_epochs - decay_after))
    return lr_lambda


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(path, G, D, opt_G, opt_D, epoch, best_ssim):
    torch.save({
        "epoch": epoch,
        "best_ssim": best_ssim,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "opt_G_state": opt_G.state_dict(),
        "opt_D_state": opt_D.state_dict(),
    }, path)


def load_checkpoint(path, G, D, opt_G, opt_D, device):
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    opt_G.load_state_dict(ckpt["opt_G_state"])
    opt_D.load_state_dict(ckpt["opt_D_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_ssim", -1.0)


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train(args):
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir,    exist_ok=True)

    # Data
    train_loader, val_loader = build_color_dataloaders(
        args.patches_dir,
        val_scenes=args.val_scenes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Models
    G = UNetGenerator(in_channels=2, out_channels=3, ngf=args.ngf).to(device)
    D = PatchGANDiscriminator(in_channels=4, ndf=args.ndf).to(device)

    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D.parameters()) / 1e6

    # Losses
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1  = nn.L1Loss()
    physics_fn    = PhysicsInformedLoss().to(device) if args.lambda_physics > 0 else None

    # Optimisers
    opt_G = Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # LR schedulers
    sched_G = LambdaLR(opt_G, make_lr_lambda(args.epochs, args.decay_after))
    sched_D = LambdaLR(opt_D, make_lr_lambda(args.epochs, args.decay_after))

    start_epoch = 0
    best_ssim   = -1.0

    # Resume
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_ssim = load_checkpoint(
            args.resume, G, D, opt_G, opt_D, device)
        log.info(f"Resumed from {args.resume}  (epoch {start_epoch}, best_ssim={best_ssim:.4f})")

    # CSV log
    log_path = os.path.join(args.results_dir, "train_log.csv")
    csv_file = open(log_path, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(["epoch", "loss_G", "loss_D", "loss_L1", "loss_GAN", "loss_physics",
                     "val_ssim", "val_l1", "lr", "time_s"])

    log.info("=" * 65)
    log.info("  Phase 3 Colorization Training -- Pix2Pix cGAN")
    log.info(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    log.info(f"  Device: {device}  |  G: {g_params:.2f}M  |  D: {d_params:.2f}M")
    log.info(f"  Train samples: {len(train_loader.dataset)}  |  Val samples: {len(val_loader.dataset)}")
    log.info(f"  lambda_L1={args.lambda_l1}  lambda_GAN={args.lambda_gan}  lambda_physics={args.lambda_physics}")
    log.info("=" * 65)

    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train()
        t0 = time.time()

        sum_loss_G = sum_loss_D = sum_loss_l1 = sum_loss_gan = sum_loss_phys = 0.0
        n_batches = 0

        for gen_in, rgb_real, _ in train_loader:
            gen_in   = gen_in.to(device)    # (B, 2, 512, 512)
            rgb_real = rgb_real.to(device)  # (B, 3, 512, 512)
            tir_cond = gen_in[:, :1]        # first channel for D

            # Forward G
            rgb_fake = G(gen_in)            # (B, 3, 512, 512)

            # ── Update Discriminator ──────────────────────────────────────────
            opt_D.zero_grad()
            loss_D = discriminator_loss(D, tir_cond, rgb_real, rgb_fake, criterion_gan)
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_D.step()

            # ── Update Generator ──────────────────────────────────────────────
            opt_G.zero_grad()
            loss_G, loss_gan, loss_l1, loss_phys = generator_loss(
                D, tir_cond, rgb_fake, rgb_real,
                criterion_gan, criterion_l1,
                args.lambda_l1, args.lambda_gan,
                physics_fn=physics_fn, lambda_physics=args.lambda_physics,
            )
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

            sum_loss_G    += loss_G.item()
            sum_loss_D    += loss_D.item()
            sum_loss_l1   += loss_l1.item()
            sum_loss_gan  += loss_gan.item()
            sum_loss_phys += loss_phys.item()
            n_batches     += 1

        sched_G.step(); sched_D.step()

        # ── Validation ────────────────────────────────────────────────────────
        G.eval()
        val_ssim = val_l1 = 0.0
        with torch.no_grad():
            for gen_in_v, rgb_real_v, _ in val_loader:
                gen_in_v   = gen_in_v.to(device)
                rgb_real_v = rgb_real_v.to(device)
                rgb_pred   = G(gen_in_v)
                val_ssim  += ssim_batch(rgb_pred, rgb_real_v)
                val_l1    += criterion_l1(rgb_pred, rgb_real_v).item()
        n_val    = len(val_loader)
        val_ssim /= n_val
        val_l1   /= n_val

        elapsed  = time.time() - t0
        avg_G    = sum_loss_G    / n_batches
        avg_D    = sum_loss_D    / n_batches
        avg_l1   = sum_loss_l1   / n_batches
        avg_gan  = sum_loss_gan  / n_batches
        avg_phys = sum_loss_phys / n_batches
        cur_lr   = opt_G.param_groups[0]["lr"]

        log.info(
            f"[Epoch {epoch+1:03d}/{args.epochs}] "
            f"loss_G={avg_G:.4f}  loss_D={avg_D:.4f}  "
            f"L1={avg_l1:.4f}  GAN={avg_gan:.4f}  phys={avg_phys:.4f}  |  "
            f"val_SSIM={val_ssim:.4f}  val_L1={val_l1:.4f}  |  "
            f"lr={cur_lr:.2e}  t={elapsed:.1f}s"
        )

        writer.writerow([epoch+1, avg_G, avg_D, avg_l1, avg_gan, avg_phys,
                         val_ssim, val_l1, cur_lr, f"{elapsed:.1f}"])
        csv_file.flush()

        # Save last checkpoint every epoch
        last_path = os.path.join(args.checkpoint_dir, "last_pix2pix.pth")
        save_checkpoint(last_path, G, D, opt_G, opt_D, epoch+1, best_ssim)

        # Save best checkpoint
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            best_path = os.path.join(args.checkpoint_dir, "best_pix2pix.pth")
            save_checkpoint(best_path, G, D, opt_G, opt_D, epoch+1, best_ssim)
            log.info(f"  => New best SSIM={best_ssim:.4f} -- checkpoint saved.")

    csv_file.close()
    log.info("=" * 65)
    log.info(f"Training complete.  Best val SSIM = {best_ssim:.4f}")
    log.info(f"Checkpoint: {os.path.join(args.checkpoint_dir, 'best_pix2pix.pth')}")
    log.info(f"Train log : {log_path}")
    log.info("=" * 65)


if __name__ == "__main__":
    train(parse_args())
