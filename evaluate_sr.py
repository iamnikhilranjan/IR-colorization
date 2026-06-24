"""
evaluate_sr.py
─────────────────────────────────────────────────────────────────────────────
Phase 2: Super-Resolution Evaluation Script

Loads the best-saved RRDBNet checkpoint and evaluates it on the validation
scenes. Computes quantitative metrics and saves side-by-side visualisations.

Usage:
    # Evaluate using the best checkpoint (default):
    python evaluate_sr.py

    # Specify checkpoint and patches directory:
    python evaluate_sr.py --checkpoint checkpoints/best_rrdbnet.pth --patches_dir output/patches

    # Evaluate on specific scenes:
    python evaluate_sr.py --val_scenes SCENE_009 SCENE_010

Output:
    output/sr_results/
    ├── metrics_summary.csv          ← per-sample PSNR / SSIM / L1 table
    ├── metrics_summary.json         ← same data as JSON for programmatic use
    ├── evaluation_report.txt        ← human-readable summary report
    └── visualisations/
        └── <scene>_<sample>_comparison.png  ← side-by-side: LR | SR | HR
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import csv
import json
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset import TIRPatchDataset, normalize_tir
from src.models.rrdbnet import build_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10((max_val ** 2) / mse)


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Full sliding-window SSIM using OpenCV — more accurate than global SSIM.
    Requires both arrays to be in [0, 1] float32.
    """
    pred_u8   = (np.clip(pred,   0, 1) * 255).astype(np.uint8)
    target_u8 = (np.clip(target, 0, 1) * 255).astype(np.uint8)

    # OpenCV SSIM (simplified; full SSIM needs additional library)
    # We use a local variance-based approximation
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    pred_f   = pred_u8.astype(np.float64)
    target_f = target_u8.astype(np.float64)

    ksize = 11
    mu1 = cv2.GaussianBlur(pred_f,   (ksize, ksize), 1.5)
    mu2 = cv2.GaussianBlur(target_f, (ksize, ksize), 1.5)

    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    s1 = cv2.GaussianBlur(pred_f   ** 2, (ksize, ksize), 1.5) - mu1_sq
    s2 = cv2.GaussianBlur(target_f ** 2, (ksize, ksize), 1.5) - mu2_sq
    s12= cv2.GaussianBlur(pred_f * target_f, (ksize, ksize), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    ssim_map = num / (den + 1e-8)
    return float(ssim_map.mean())


def compute_l1(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


# ── Visualisation ─────────────────────────────────────────────────────────

def percentile_stretch(img: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    """Stretch a float32 array to uint8 for visualisation."""
    lo_v = np.percentile(img, lo)
    hi_v = np.percentile(img, hi)
    stretched = np.clip(img, lo_v, hi_v)
    denom = max(hi_v - lo_v, 1e-5)
    return ((stretched - lo_v) / denom * 255).astype(np.uint8)


def make_comparison_image(
    lr_np: np.ndarray,
    sr_np: np.ndarray,
    hr_np: np.ndarray,
    psnr_val: float,
    ssim_val: float,
    l1_val: float,
) -> np.ndarray:
    """
    Create a side-by-side comparison image:
    [LR@200m (256px upsampled) | SR Output (512px) | HR@100m GT (512px)]

    The LR image is bilinearly upsampled to 512 for visual comparison only.
    """
    TARGET = 512
    GAP    = 8
    LABEL_H = 32

    def to_bgr(arr):
        """Convert 2D or (1,H,W) float array to BGR uint8 at TARGET resolution."""
        if arr.ndim == 3:
            arr = arr[0]
        vis = percentile_stretch(arr.astype(np.float32))
        vis = cv2.resize(vis, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    col_lr = to_bgr(lr_np)
    col_sr = to_bgr(sr_np)
    col_hr = to_bgr(hr_np)

    gap = np.full((TARGET, GAP, 3), 30, dtype=np.uint8)
    row = np.hstack([col_lr, gap, col_sr, gap, col_hr])

    # Label bar
    label = np.full((LABEL_H, row.shape[1], 3), 20, dtype=np.uint8)
    texts = [
        "LR Input (TIR@200m — bicubic up)",
        f"SR Output (RRDBNet x2)   PSNR={psnr_val:.2f}dB  SSIM={ssim_val:.4f}",
        "HR Ground Truth (TIR@100m)",
    ]
    x_pos = [4, TARGET + GAP + 4, 2 * (TARGET + GAP) + 4]
    for txt, xp in zip(texts, x_pos):
        cv2.putText(label, txt, (xp, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 220, 255), 1, cv2.LINE_AA)

    # Metric bar below
    metric_bar = np.full((28, row.shape[1], 3), 15, dtype=np.uint8)
    metric_txt = (
        f"L1 Error={l1_val:.5f}   |   PSNR={psnr_val:.2f} dB   |   SSIM={ssim_val:.4f}"
    )
    cv2.putText(metric_bar, metric_txt, (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (100, 255, 150), 1, cv2.LINE_AA)

    return np.vstack([label, row, metric_bar])


# ── Main evaluation ───────────────────────────────────────────────────────

def evaluate(args):
    # ── Device ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluation device: {device}")

    # ── Check checkpoint ──
    if not os.path.isfile(args.checkpoint):
        logger.error(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Train first: python train_sr.py"
        )
        sys.exit(1)

    # ── Model ──
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Infer architecture from saved args (if available)
    saved_args = ckpt.get("args", {})
    num_block = saved_args.get("num_block", args.num_block)
    num_feat  = saved_args.get("num_feat",  args.num_feat)

    model = build_model(num_block=num_block, num_feat=num_feat, scale=2, device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Loaded model from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # ── Dataset (validation scenes only, no augmentation) ──
    val_ds = TIRPatchDataset(
        patches_root=args.patches_dir,
        scene_ids=args.val_scenes,
        augment=False,
    )

    if len(val_ds) == 0:
        logger.error(
            f"No samples found in validation scenes {args.val_scenes}. "
            "Check --patches_dir and --val_scenes."
        )
        sys.exit(1)

    logger.info(f"Evaluating {len(val_ds)} samples from {args.val_scenes}")

    # ── Output directories ──
    vis_dir = os.path.join(args.out_dir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)

    # ── Metrics storage ──
    all_metrics = []

    with torch.no_grad():
        for idx in range(len(val_ds)):
            sample_dir = val_ds.samples[idx]
            scene_id   = Path(sample_dir).parent.name
            sample_id  = Path(sample_dir).name

            # Load originals directly for accurate denormalization (visualisation)
            lr_raw = np.load(os.path.join(sample_dir, "tir_200m.npy")).astype(np.float32)
            hr_raw = np.load(os.path.join(sample_dir, "tir_100m_512.npy")).astype(np.float32)

            # Normalize (same as training)
            lr_norm, _, _ = normalize_tir(lr_raw)
            hr_norm, _, _ = normalize_tir(hr_raw)

            # Ensure (1, H, W)
            def to_tensor(arr):
                if arr.ndim == 2:
                    arr = arr[np.newaxis]
                elif arr.ndim == 3:
                    arr = arr[:1]
                return torch.from_numpy(arr).unsqueeze(0).to(device)

            lr_t = to_tensor(lr_norm)
            hr_t = to_tensor(hr_norm)

            # SR forward pass
            sr_t = model(lr_t)
            sr_t = torch.clamp(sr_t, 0.0, 1.0)

            # To numpy for metrics
            sr_np = sr_t.squeeze().cpu().numpy()
            hr_np = hr_t.squeeze().cpu().numpy()
            lr_np = lr_t.squeeze().cpu().numpy()

            p = compute_psnr(sr_np, hr_np)
            s = compute_ssim(sr_np, hr_np)
            l = compute_l1(sr_np, hr_np)

            all_metrics.append({
                "scene":   scene_id,
                "sample":  sample_id,
                "psnr_db": round(p, 4),
                "ssim":    round(s, 4),
                "l1":      round(l, 6),
            })

            logger.info(
                f"  {scene_id}/{sample_id}  PSNR={p:.2f} dB  SSIM={s:.4f}  L1={l:.5f}"
            )

            # Save visualisation
            if args.save_vis:
                vis = make_comparison_image(lr_np, sr_np, hr_np, p, s, l)
                vis_path = os.path.join(vis_dir, f"{scene_id}_{sample_id}_comparison.png")
                cv2.imwrite(vis_path, vis)

    # ── Summary statistics ──
    psnr_vals = [m["psnr_db"] for m in all_metrics]
    ssim_vals = [m["ssim"]    for m in all_metrics]
    l1_vals   = [m["l1"]      for m in all_metrics]

    summary = {
        "total_samples": len(all_metrics),
        "mean_psnr_db":  round(float(np.mean(psnr_vals)), 4),
        "std_psnr_db":   round(float(np.std(psnr_vals)),  4),
        "mean_ssim":     round(float(np.mean(ssim_vals)), 4),
        "std_ssim":      round(float(np.std(ssim_vals)),  4),
        "mean_l1":       round(float(np.mean(l1_vals)),   6),
        "checkpoint":    args.checkpoint,
        "evaluated_at":  datetime.now().isoformat(),
    }

    # ── Save CSV ──
    csv_path = os.path.join(args.out_dir, "metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "sample", "psnr_db", "ssim", "l1"])
        writer.writeheader()
        writer.writerows(all_metrics)
    logger.info(f"Metrics CSV saved: {csv_path}")

    # ── Save JSON ──
    json_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "samples": all_metrics}, f, indent=2)
    logger.info(f"Metrics JSON saved: {json_path}")

    # ── Save human-readable report ──
    report_path = os.path.join(args.out_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  Phase 2 SR — Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Val Scenes : {args.val_scenes}\n")
        f.write(f"Total Samples: {summary['total_samples']}\n\n")
        f.write("─" * 60 + "\n")
        f.write("  Quantitative Metrics\n")
        f.write("─" * 60 + "\n")
        f.write(f"  PSNR : {summary['mean_psnr_db']:.2f} ± {summary['std_psnr_db']:.2f} dB\n")
        f.write(f"  SSIM : {summary['mean_ssim']:.4f} ± {summary['std_ssim']:.4f}\n")
        f.write(f"  L1   : {summary['mean_l1']:.6f}\n\n")
        f.write("─" * 60 + "\n")
        f.write("  Per-Sample Breakdown\n")
        f.write("─" * 60 + "\n")
        for m in all_metrics:
            f.write(
                f"  {m['scene']}/{m['sample']}: "
                f"PSNR={m['psnr_db']:.2f} dB  SSIM={m['ssim']:.4f}  L1={m['l1']:.5f}\n"
            )
        f.write("\n")
        f.write("=" * 60 + "\n")

    # ── Print summary ──
    logger.info("\n" + "=" * 60)
    logger.info("  EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Samples evaluated : {summary['total_samples']}")
    logger.info(f"  Mean PSNR         : {summary['mean_psnr_db']:.2f} ± {summary['std_psnr_db']:.2f} dB")
    logger.info(f"  Mean SSIM         : {summary['mean_ssim']:.4f} ± {summary['std_ssim']:.4f}")
    logger.info(f"  Mean L1           : {summary['mean_l1']:.6f}")
    logger.info("=" * 60)
    if args.save_vis:
        logger.info(f"  Visualisations    : {vis_dir}/")
    logger.info(f"  Report            : {report_path}")
    logger.info("=" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: Evaluate TIR Super-Resolution (PSNR / SSIM)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_rrdbnet.pth",
        help="Path to the trained model checkpoint (default: checkpoints/best_rrdbnet.pth)"
    )
    parser.add_argument(
        "--patches_dir", type=str, default="output/patches",
        help="Path to the output/patches directory (default: output/patches)"
    )
    parser.add_argument(
        "--val_scenes", type=str, nargs="+",
        default=["SCENE_009", "SCENE_010"],
        help="Scene IDs to evaluate on (default: SCENE_009 SCENE_010)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="output/sr_results",
        help="Directory to save evaluation outputs (default: output/sr_results)"
    )
    parser.add_argument(
        "--save_vis", action="store_true", default=True,
        help="Save side-by-side comparison images (default: True)"
    )
    parser.add_argument(
        "--num_block", type=int, default=6,
        help="RRDB blocks (overridden by saved checkpoint args if available)"
    )
    parser.add_argument(
        "--num_feat", type=int, default=64,
        help="Feature channels (overridden by saved checkpoint args if available)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
