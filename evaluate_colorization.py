"""
evaluate_colorization.py  --  Phase 3 Evaluation
Loads best_pix2pix.pth and produces:
  - Per-sample PSNR, SSIM, L1
  - Side-by-side comparison PNGs: TIR input | Predicted RGB | Ground Truth RGB
  - metrics_summary.csv + evaluation_report.txt

Usage:
    python evaluate_colorization.py
    python evaluate_colorization.py --checkpoint checkpoints/best_pix2pix.pth --save_vis
"""

import os
import sys
import csv
import json
import logging
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.pix2pix import UNetGenerator
from src.colorization_dataset import ColorizationDataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 Colorization Evaluation")
    p.add_argument("--checkpoint",  default="checkpoints/best_pix2pix.pth")
    p.add_argument("--patches_dir", default="output/patches")
    p.add_argument("--val_scenes",  nargs="+", default=["SCENE_009", "SCENE_010"])
    p.add_argument("--results_dir", default="output/colorization_results")
    p.add_argument("--ngf",         type=int, default=64)
    p.add_argument("--save_vis",    action="store_true", default=True,
                   help="Save side-by-side comparison images")
    p.add_argument("--device",      default="auto")
    return p.parse_args()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def psnr(pred, target, max_val=1.0):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val**2 / mse)


def ssim_np(pred, target):
    """Simple channel-averaged SSIM on [0,1] arrays."""
    scores = []
    for c in range(pred.shape[0]):
        p, t = pred[c].astype(np.float64), target[c].astype(np.float64)
        mu1, mu2 = p.mean(), t.mean()
        s1, s2   = p.std(), t.std()
        cov      = ((p - mu1) * (t - mu2)).mean()
        c1, c2   = 0.01**2, 0.03**2
        scores.append(
            ((2*mu1*mu2 + c1) * (2*cov + c2)) /
            ((mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2))
        )
    return float(np.mean(scores))


# ─── Visualisation ────────────────────────────────────────────────────────────

def make_comparison_image(tir_norm, rgb_pred, rgb_gt, scene_id, psnr_v, ssim_v):
    """
    Returns an HxW*3 side-by-side BGR image:
       [TIR input (gray→colormap)] | [Predicted RGB] | [Ground Truth RGB]
    """
    H, W = tir_norm.shape[-2], tir_norm.shape[-1]

    # TIR panel: apply thermal colormap
    tir_u8 = (tir_norm[0] * 255).clip(0, 255).astype(np.uint8)
    tir_col = cv2.applyColorMap(tir_u8, cv2.COLORMAP_INFERNO)   # (H,W,3) BGR

    def to_bgr(arr_chw):
        """(3,H,W) float [0,1] -> (H,W,3) uint8 BGR"""
        arr = np.clip(arr_chw, 0, 1)
        arr_hwc = (arr * 255).astype(np.uint8).transpose(1, 2, 0)  # (H,W,3) RGB
        return cv2.cvtColor(arr_hwc, cv2.COLOR_RGB2BGR)

    pred_bgr = to_bgr(rgb_pred)
    gt_bgr   = to_bgr(rgb_gt)

    panel = np.concatenate([tir_col, pred_bgr, gt_bgr], axis=1)  # (H, W*3, 3)

    # Add header bar
    header_h = 50
    header = np.zeros((header_h, panel.shape[1], 3), dtype=np.uint8)
    labels = [
        (f"TIR@100m Input",          W//2),
        (f"Predicted RGB  PSNR={psnr_v:.2f}dB  SSIM={ssim_v:.4f}", W + W//2),
        (f"Ground Truth RGB",         2*W + W//2),
    ]
    for text, cx in labels:
        tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(header, text, (cx - tw//2, header_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Footer with metrics
    footer_h = 40
    footer = np.zeros((footer_h, panel.shape[1], 3), dtype=np.uint8)
    foot_text = f"L1 Error = {np.mean(np.abs(rgb_pred - rgb_gt)):.5f}  |  {scene_id}"
    cv2.putText(footer, foot_text, (20, footer_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)

    return np.concatenate([header, panel, footer], axis=0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    log.info(f"Evaluation device: {device}")

    # Load model
    G = UNetGenerator(in_channels=2, out_channels=3, ngf=args.ngf).to(device)
    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    log.info(f"[UNetGenerator] Parameters: {g_params:.3f}M")

    if not os.path.exists(args.checkpoint):
        log.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    epoch_saved = ckpt.get("epoch", "?")
    log.info(f"Loaded generator from {args.checkpoint}  (epoch {epoch_saved})")

    # Dataset
    val_ds = ColorizationDataset(
        args.patches_dir, scenes=args.val_scenes, augment=False)
    log.info(f"Evaluating {len(val_ds)} samples from {args.val_scenes}")

    os.makedirs(args.results_dir, exist_ok=True)
    vis_dir = os.path.join(args.results_dir, "visualisations")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    # Evaluate
    G.eval()
    results = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            gen_in, rgb_target, scene_id = val_ds[i]
            gen_in_t    = gen_in.unsqueeze(0).to(device)   # (1,2,512,512)
            rgb_pred_t  = G(gen_in_t).squeeze(0)           # (3,512,512) in [-1,1]

            # Convert to [0,1] numpy
            rgb_pred_np = ((rgb_pred_t.cpu().numpy() + 1) / 2).clip(0, 1)
            rgb_gt_np   = ((rgb_target.numpy()        + 1) / 2).clip(0, 1)
            tir_norm_np = gen_in.numpy()                   # (2,512,512), ch0=TIR

            psnr_v = psnr(rgb_pred_np, rgb_gt_np)
            ssim_v = ssim_np(rgb_pred_np, rgb_gt_np)
            l1_v   = float(np.mean(np.abs(rgb_pred_np - rgb_gt_np)))

            log.info(f"  {scene_id:40s}  PSNR={psnr_v:.2f} dB  SSIM={ssim_v:.4f}  L1={l1_v:.5f}")
            results.append({
                "scene": scene_id, "psnr": psnr_v, "ssim": ssim_v, "l1": l1_v
            })

            # Save comparison image
            if args.save_vis:
                comp = make_comparison_image(
                    tir_norm_np, rgb_pred_np, rgb_gt_np, scene_id, psnr_v, ssim_v)
                fname = scene_id.replace("/", "_") + "_color_comparison.png"
                cv2.imwrite(os.path.join(vis_dir, fname), comp)

    # Aggregate
    psnr_vals = [r["psnr"] for r in results]
    ssim_vals = [r["ssim"] for r in results]
    l1_vals   = [r["l1"]   for r in results]

    mean_psnr = np.mean(psnr_vals);  std_psnr = np.std(psnr_vals)
    mean_ssim = np.mean(ssim_vals);  std_ssim = np.std(ssim_vals)
    mean_l1   = np.mean(l1_vals)

    # Save CSV
    csv_path = os.path.join(args.results_dir, "metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "psnr", "ssim", "l1"])
        w.writeheader(); w.writerows(results)
    log.info(f"Metrics CSV saved: {csv_path}")

    # Save JSON
    json_path = os.path.join(args.results_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump({
            "samples": results,
            "aggregate": {"mean_psnr": mean_psnr, "mean_ssim": mean_ssim, "mean_l1": mean_l1}
        }, f, indent=2)

    # Save report
    report_path = os.path.join(args.results_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  PHASE 3 COLORIZATION EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint : {args.checkpoint}  (epoch {epoch_saved})\n")
        f.write(f"Samples    : {len(results)}\n\n")
        for r in results:
            f.write(f"  {r['scene']:40s}  PSNR={r['psnr']:.2f} dB  "
                    f"SSIM={r['ssim']:.4f}  L1={r['l1']:.5f}\n")
        f.write("\n" + "-"*60 + "\n")
        f.write(f"  Mean PSNR : {mean_psnr:.2f} +/- {std_psnr:.2f} dB\n")
        f.write(f"  Mean SSIM : {mean_ssim:.4f} +/- {std_ssim:.4f}\n")
        f.write(f"  Mean L1   : {mean_l1:.6f}\n")
        f.write("=" * 60 + "\n")

    log.info("")
    log.info("=" * 60)
    log.info("  COLORIZATION EVALUATION COMPLETE")
    log.info("=" * 60)
    log.info(f"  Samples evaluated : {len(results)}")
    log.info(f"  Mean PSNR         : {mean_psnr:.2f} +/- {std_psnr:.2f} dB")
    log.info(f"  Mean SSIM         : {mean_ssim:.4f} +/- {std_ssim:.4f}")
    log.info(f"  Mean L1           : {mean_l1:.6f}")
    if args.save_vis:
        log.info(f"  Visualisations    : {vis_dir}/")
    log.info(f"  Report            : {report_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    evaluate(parse_args())
