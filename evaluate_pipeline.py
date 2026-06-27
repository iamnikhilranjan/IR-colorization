"""
evaluate_pipeline.py  --  Phase 4: Full End-to-End Pipeline Evaluation

Runs the complete chain on validation scenes:
  TIR@200m -> [SR] -> TIR@100m -> [Colorize] -> RGB@100m

Compares predicted RGB against ground-truth rgb_100m_512.npy.
Also measures per-stage inference time.

Usage:
    python evaluate_pipeline.py
    python evaluate_pipeline.py --sr_checkpoint checkpoints/best_rrdbnet.pth \
                                 --color_checkpoint checkpoints/best_pix2pix.pth
"""

import os, sys, csv, json, time, logging, argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pipeline import IRColorizationPipeline

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4: Full Pipeline Evaluation")
    p.add_argument("--sr_checkpoint",    default="checkpoints/best_rrdbnet.pth")
    p.add_argument("--color_checkpoint", default="checkpoints/best_pix2pix.pth")
    p.add_argument("--patches_dir",      default="output/patches")
    p.add_argument("--val_scenes", nargs="+", default=["SCENE_009", "SCENE_010"])
    p.add_argument("--results_dir",      default="output/pipeline_results")
    p.add_argument("--num_blocks",  type=int, default=6)
    p.add_argument("--ngf",         type=int, default=64)
    p.add_argument("--device",           default="auto")
    p.add_argument("--save_vis",    action="store_true", default=True)
    return p.parse_args()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim_np(pred, target):
    scores = []
    for c in range(pred.shape[0]):
        p, t = pred[c].astype(np.float64), target[c].astype(np.float64)
        mu1, mu2 = p.mean(), t.mean()
        s1, s2   = p.std(),  t.std()
        cov      = ((p - mu1) * (t - mu2)).mean()
        c1, c2   = 0.01**2, 0.03**2
        scores.append(
            ((2*mu1*mu2 + c1)*(2*cov + c2)) /
            ((mu1**2+mu2**2+c1)*(s1**2+s2**2+c2))
        )
    return float(np.mean(scores))


def percentile_norm(arr, lo_pct=1, hi_pct=99):
    lo, hi = np.percentile(arr, lo_pct), np.percentile(arr, hi_pct)
    if hi > lo:
        return np.clip((arr - lo)/(hi - lo), 0, 1).astype(np.float32)
    return np.zeros_like(arr, dtype=np.float32)


# ─── Visualisation ────────────────────────────────────────────────────────────

def save_comparison(path, tir_lr, tir_sr, rgb_pred, rgb_gt, meta):
    H, W = 512, 512

    def to_bgr_u8(arr_chw):
        arr = np.clip(arr_chw, 0, 1)
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        hwc = (arr * 255).astype(np.uint8).transpose(1, 2, 0)
        return cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)

    # Resize LR TIR to 512 for display
    tir_lr_norm = percentile_norm(tir_lr)
    tir_lr_up   = cv2.resize(tir_lr_norm[0], (W, H), interpolation=cv2.INTER_NEAREST)
    tir_lr_col  = cv2.applyColorMap((tir_lr_up*255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    tir_sr_col  = cv2.applyColorMap(
        (np.clip(tir_sr[0], 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    pred_bgr = to_bgr_u8(rgb_pred)
    gt_bgr   = to_bgr_u8(rgb_gt)

    panel = np.concatenate([tir_lr_col, tir_sr_col, pred_bgr, gt_bgr], axis=1)

    # Header
    hdr = np.zeros((55, panel.shape[1], 3), np.uint8)
    labels = [
        ("TIR@200m Input",            W//2),
        ("SR TIR@100m (RRDBNet)",     W + W//2),
        (f"Pred RGB  PSNR={meta['psnr']:.2f}dB  SSIM={meta['ssim']:.3f}", 2*W + W//2),
        ("GT RGB@100m",               3*W + W//2),
    ]
    for txt, cx in labels:
        tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        cv2.putText(hdr, txt, (cx - tw//2, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    ftr = np.zeros((40, panel.shape[1], 3), np.uint8)
    cv2.putText(ftr,
        f"L1={meta['l1']:.5f}  SR_time={meta['sr_ms']:.1f}ms  "
        f"Color_time={meta['color_ms']:.1f}ms  |  {meta['scene']}",
        (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)

    cv2.imwrite(path, np.concatenate([hdr, panel, ftr], axis=0))


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate(args):
    os.makedirs(args.results_dir, exist_ok=True)
    vis_dir = os.path.join(args.results_dir, "visualisations")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    color_ckpt = args.color_checkpoint if os.path.exists(args.color_checkpoint) else None
    pipe = IRColorizationPipeline(
        sr_checkpoint    = args.sr_checkpoint,
        color_checkpoint = color_ckpt,
        device           = args.device,
        num_blocks       = args.num_blocks,
        ngf              = args.ngf,
    )

    # Collect val samples
    samples = []
    for scene in sorted(os.listdir(args.patches_dir)):
        if scene not in args.val_scenes:
            continue
        scene_dir = os.path.join(args.patches_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        for samp in sorted(os.listdir(scene_dir)):
            samp_dir = os.path.join(scene_dir, samp)
            tir_lr_p = os.path.join(samp_dir, "tir_200m.npy")
            rgb_gt_p = os.path.join(samp_dir, "rgb_100m_512.npy")
            tir_hr_p = os.path.join(samp_dir, "tir_100m_512.npy")
            if os.path.exists(tir_lr_p) and os.path.exists(rgb_gt_p):
                samples.append((tir_lr_p, rgb_gt_p, tir_hr_p, f"{scene}/{samp}"))

    log.info(f"Evaluating {len(samples)} samples  |  device={pipe.device}")
    results = []

    for tir_lr_p, rgb_gt_p, tir_hr_p, scene_id in samples:
        tir_lr = np.load(tir_lr_p).astype(np.float32)
        rgb_gt = np.load(rgb_gt_p).astype(np.float32)
        if tir_lr.ndim == 2: tir_lr = tir_lr[np.newaxis]
        if rgb_gt.ndim == 2: rgb_gt = rgb_gt[np.newaxis]

        # Normalize GT RGB
        rgb_gt_norm = np.zeros_like(rgb_gt)
        for c in range(rgb_gt.shape[0]):
            rgb_gt_norm[c] = percentile_norm(rgb_gt[c])

        # ── Stage 1: SR ──────────────────────────────────────────────────────
        t0      = time.perf_counter()
        tir_sr  = pipe.run_sr(tir_lr)
        sr_ms   = (time.perf_counter() - t0) * 1000

        # ── Stage 2: Colorize ─────────────────────────────────────────────────
        color_ms = 0.0
        if pipe.colorizer is not None:
            t0       = time.perf_counter()
            rgb_pred = pipe.run_colorization(tir_sr)
            color_ms = (time.perf_counter() - t0) * 1000
        else:
            rgb_pred = np.repeat(tir_sr, 3, axis=0)
            log.warning("Colorizer not loaded — using SR grayscale as RGB.")

        # ── Metrics ───────────────────────────────────────────────────────────
        psnr_v = psnr(rgb_pred, rgb_gt_norm)
        ssim_v = ssim_np(rgb_pred, rgb_gt_norm)
        l1_v   = float(np.mean(np.abs(rgb_pred - rgb_gt_norm)))

        meta = dict(scene=scene_id, psnr=psnr_v, ssim=ssim_v,
                    l1=l1_v, sr_ms=sr_ms, color_ms=color_ms)
        results.append(meta)

        log.info(
            f"  {scene_id:40s}  PSNR={psnr_v:.2f}dB  SSIM={ssim_v:.4f}  "
            f"L1={l1_v:.5f}  SR={sr_ms:.1f}ms  Color={color_ms:.1f}ms"
        )

        if args.save_vis:
            fname = scene_id.replace("/", "_") + "_pipeline.png"
            save_comparison(
                os.path.join(vis_dir, fname),
                tir_lr, tir_sr, rgb_pred, rgb_gt_norm, meta
            )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    mean_psnr  = np.mean([r["psnr"]     for r in results])
    mean_ssim  = np.mean([r["ssim"]     for r in results])
    mean_l1    = np.mean([r["l1"]       for r in results])
    mean_sr    = np.mean([r["sr_ms"]    for r in results])
    mean_color = np.mean([r["color_ms"] for r in results])
    total_ms   = mean_sr + mean_color

    # Save CSV
    csv_path = os.path.join(args.results_dir, "pipeline_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)

    # Save JSON
    json_path = os.path.join(args.results_dir, "pipeline_metrics.json")
    with open(json_path, "w") as f:
        json.dump({
            "samples": results,
            "aggregate": {
                "mean_psnr": mean_psnr, "mean_ssim": mean_ssim,
                "mean_l1": mean_l1, "mean_sr_ms": mean_sr,
                "mean_color_ms": mean_color, "total_inference_ms": total_ms
            }
        }, f, indent=2)

    # Save report
    report_path = os.path.join(args.results_dir, "pipeline_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("  PHASE 4: FULL PIPELINE EVALUATION REPORT\n")
        f.write("  TIR@200m -> SR -> Colorize -> RGB@100m\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"SR  checkpoint   : {args.sr_checkpoint}\n")
        f.write(f"Col checkpoint   : {args.color_checkpoint}\n")
        f.write(f"Samples evaluated: {len(results)}\n\n")
        for r in results:
            f.write(f"  {r['scene']:40s}  PSNR={r['psnr']:.2f}dB  "
                    f"SSIM={r['ssim']:.4f}  L1={r['l1']:.5f}  "
                    f"SR={r['sr_ms']:.1f}ms  Color={r['color_ms']:.1f}ms\n")
        f.write("\n" + "-"*65 + "\n")
        f.write(f"  Mean PSNR        : {mean_psnr:.2f} dB\n")
        f.write(f"  Mean SSIM        : {mean_ssim:.4f}\n")
        f.write(f"  Mean L1          : {mean_l1:.6f}\n")
        f.write(f"  Avg SR time      : {mean_sr:.1f} ms / tile\n")
        f.write(f"  Avg Color time   : {mean_color:.1f} ms / tile\n")
        f.write(f"  Total inference  : {total_ms:.1f} ms / tile  "
                f"({1000/total_ms:.1f} tiles/sec)\n")
        f.write("=" * 65 + "\n")

    log.info("")
    log.info("=" * 65)
    log.info("  PHASE 4 PIPELINE EVALUATION COMPLETE")
    log.info("=" * 65)
    log.info(f"  Samples evaluated  : {len(results)}")
    log.info(f"  Mean PSNR          : {mean_psnr:.2f} dB")
    log.info(f"  Mean SSIM          : {mean_ssim:.4f}")
    log.info(f"  Mean L1            : {mean_l1:.6f}")
    log.info(f"  Avg SR time        : {mean_sr:.1f} ms/tile")
    log.info(f"  Avg Color time     : {mean_color:.1f} ms/tile")
    log.info(f"  Total inference    : {total_ms:.1f} ms/tile  ({1000/total_ms:.1f} tiles/sec)")
    if args.save_vis:
        log.info(f"  Visualisations     : {vis_dir}/")
    log.info(f"  Report             : {report_path}")
    log.info("=" * 65)


if __name__ == "__main__":
    evaluate(parse_args())
