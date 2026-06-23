"""
scripts/verify_patches.py
─────────────────────────────────────────────────────────────────────────────
Verify the spatial alignment and shape correctness of generated patches,
and save a grid of visual previews.

Usage
─────
    python scripts/verify_patches.py --patches_dir output/patches
    python scripts/verify_patches.py --patches_dir output/patches --save_grid

Output
──────
  • Console report: per-sample pass/fail with shape info
  • output/patch_previews/<product_id>_preview.png  (8-patch grid per scene)
  • output/patch_previews/summary.json              (machine-readable stats)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import json
import glob
import logging

import numpy as np
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Expected shapes ────────────────────────────────────────────────────────

EXPECTED = {
    "tir_200m":     {"suffix": "tir_200m.npy",     "spatial": (256, 256)},
    "tir_100m_512": {"suffix": "tir_100m_512.npy", "spatial": (512, 512)},
    "rgb_100m_512": {"suffix": "rgb_100m_512.npy", "spatial": (512, 512)},
}


# ── Alignment check ────────────────────────────────────────────────────────

def check_alignment(tir_200m: np.ndarray, tir_100m: np.ndarray) -> bool:
    """
    Each pixel in tir_200m must correspond to a 2×2 block in tir_100m.
    We verify by downsampling tir_100m by 2 and computing mean absolute error
    against tir_200m (both rescaled to [0,1]).
    """
    tir_200m_2d = tir_200m if tir_200m.ndim == 2 else tir_200m[0]
    tir_100m_2d = tir_100m if tir_100m.ndim == 2 else tir_100m[0]

    # Downscale 512→256 using box average
    h, w = tir_200m_2d.shape
    tir_100m_downscaled = cv2.resize(
        tir_100m_2d.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_AREA,
    )

    # Normalise both
    def norm(x):
        xmin, xmax = x.min(), x.max()
        if xmax == xmin:
            return np.zeros_like(x, dtype=np.float32)
        return (x - xmin) / (xmax - xmin + 1e-8)

    mae = np.mean(np.abs(norm(tir_200m_2d.astype(np.float32)) - norm(tir_100m_downscaled)))
    return mae < 0.15  # Threshold: <15% MAE accepted


# ── Visual helpers ─────────────────────────────────────────────────────────

def percentile_stretch(img: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Stretch a single-channel or 3-channel uint8 image."""
    if img.ndim == 3 and img.shape[0] <= 4:  # (C, H, W) → (H, W, C)
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 3:
        out = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[-1]):
            out[..., c] = _stretch_channel(img[..., c], lo, hi)
        return out.astype(np.uint8)
    return _stretch_channel(img, lo, hi).astype(np.uint8)


def _stretch_channel(ch: np.ndarray, lo: float, hi: float) -> np.ndarray:
    lo_v, hi_v = np.percentile(ch, lo), np.percentile(ch, hi)
    stretched = np.clip(ch.astype(np.float32), lo_v, hi_v)
    denom = max(hi_v - lo_v, 1e-5)
    return (stretched - lo_v) * 255.0 / denom


def make_preview_row(sample_dir: str) -> np.ndarray | None:
    """
    Build a single horizontal row of 3 side-by-side patches:
      [TIR@200m 256px] [TIR@100m → resized to 256px] [RGB@100m → resized to 256px]
    Returns an (256, 3*256+2*gap, 3) BGR image, or None on failure.
    """
    TARGET = 256
    GAP = 4

    def load(name):
        path = os.path.join(sample_dir, EXPECTED[name]["suffix"])
        if not os.path.exists(path):
            return None
        return np.load(path)

    tir_200 = load("tir_200m")
    tir_100 = load("tir_100m_512")
    rgb_100 = load("rgb_100m_512")

    if any(x is None for x in [tir_200, tir_100, rgb_100]):
        return None

    def to_bgr_256(arr):
        """Convert any array to BGR uint8 at TARGET×TARGET."""
        vis = percentile_stretch(arr)
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        elif vis.ndim == 3 and vis.shape[-1] == 1:
            vis = cv2.cvtColor(vis[..., 0], cv2.COLOR_GRAY2BGR)
        elif vis.ndim == 3 and vis.shape[-1] == 3:
            pass  # already BGR (or RGB — fine for preview)
        elif vis.ndim == 3 and vis.shape[-1] > 3:
            vis = vis[..., :3]
        if vis.shape[0] != TARGET or vis.shape[1] != TARGET:
            vis = cv2.resize(vis, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)
        return vis

    col1 = to_bgr_256(tir_200)
    col2 = to_bgr_256(tir_100)
    col3 = to_bgr_256(rgb_100)

    gap = np.full((TARGET, GAP, 3), 40, dtype=np.uint8)
    row = np.hstack([col1, gap, col2, gap, col3])

    # Add label bar
    label_h = 20
    label = np.full((label_h, row.shape[1], 3), 20, dtype=np.uint8)
    labels = ["TIR@200m (input)", "TIR@100m (SR target)", "RGB@100m (color target)"]
    x_positions = [0, TARGET + GAP, 2 * (TARGET + GAP)]
    for txt, xp in zip(labels, x_positions):
        cv2.putText(label, txt, (xp + 4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (200, 200, 200), 1, cv2.LINE_AA)

    return np.vstack([label, row])


# ── Main verification logic ────────────────────────────────────────────────

def verify_patches(patches_dir: str, save_grid: bool, preview_dir: str) -> dict:
    """Walk patches_dir and validate every sample. Returns a summary dict."""
    if not os.path.isdir(patches_dir):
        logger.error(f"patches_dir does not exist: {patches_dir}")
        sys.exit(1)

    os.makedirs(preview_dir, exist_ok=True)

    products = sorted(
        e for e in os.listdir(patches_dir)
        if os.path.isdir(os.path.join(patches_dir, e))
    )

    if not products:
        logger.warning("No product folders found in patches_dir.")
        return {}

    overall_pass = 0
    overall_fail = 0
    summary = {}

    for product_id in products:
        product_dir = os.path.join(patches_dir, product_id)
        sample_dirs = sorted(glob.glob(os.path.join(product_dir, "sample_*")))

        product_stats = {"total": len(sample_dirs), "pass": 0, "fail": 0, "errors": []}
        preview_rows = []

        logger.info(f"\n{'─'*60}")
        logger.info(f"Product: {product_id}  ({len(sample_dirs)} samples)")

        for sd in sample_dirs:
            sample_name = os.path.basename(sd)
            errors = []

            # 1. Shape checks
            arrays = {}
            for key, spec in EXPECTED.items():
                fpath = os.path.join(sd, spec["suffix"])
                if not os.path.exists(fpath):
                    errors.append(f"Missing {spec['suffix']}")
                    continue
                arr = np.load(fpath)
                arrays[key] = arr
                spatial = arr.shape[-2:]
                if tuple(spatial) != spec["spatial"]:
                    errors.append(
                        f"{key}: expected spatial {spec['spatial']}, got {spatial}"
                    )

            # 2. Non-zero check
            for key, arr in arrays.items():
                if arr.max() == 0:
                    errors.append(f"{key}: all-zero array (likely bad download)")

            # 3. Alignment check (TIR 200m vs TIR 100m)
            if "tir_200m" in arrays and "tir_100m_512" in arrays:
                if not check_alignment(arrays["tir_200m"], arrays["tir_100m_512"]):
                    errors.append("Alignment: TIR@200m vs TIR@100m MAE too high")

            if errors:
                product_stats["fail"] += 1
                overall_fail += 1
                product_stats["errors"].append({sample_name: errors})
                logger.warning(f"  ✗ {sample_name}: {errors}")
            else:
                product_stats["pass"] += 1
                overall_pass += 1
                logger.debug(f"  ✓ {sample_name}")

            # Build preview (first 8 samples)
            if save_grid and len(preview_rows) < 8:
                row_img = make_preview_row(sd)
                if row_img is not None:
                    preview_rows.append(row_img)

        # Save preview grid for this product
        if save_grid and preview_rows:
            grid = np.vstack(preview_rows)
            grid_path = os.path.join(preview_dir, f"{product_id}_preview.png")
            cv2.imwrite(grid_path, grid)
            logger.info(f"  Preview saved → {grid_path}")

        summary[product_id] = product_stats
        logger.info(
            f"  Result: {product_stats['pass']}/{product_stats['total']} passed"
        )

    # Write machine-readable summary
    summary_path = os.path.join(preview_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═'*60}")
    print(f"Verification Complete")
    print(f"  Total PASS : {overall_pass}")
    print(f"  Total FAIL : {overall_fail}")
    print(f"  Summary    : {summary_path}")
    if save_grid:
        print(f"  Previews   : {preview_dir}/")
    print(f"{'═'*60}\n")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify patch alignment and shapes after driver.py pipeline."
    )
    parser.add_argument(
        "--patches_dir",
        type=str,
        default="output/patches",
        help="Path to patches output dir (default: output/patches)",
    )
    parser.add_argument(
        "--preview_dir",
        type=str,
        default="output/patch_previews",
        help="Where to write preview PNGs (default: output/patch_previews)",
    )
    parser.add_argument(
        "--save_grid",
        action="store_true",
        default=True,
        help="Save preview grids (default: True). Use --no_save_grid to disable.",
    )
    parser.add_argument("--no_save_grid", dest="save_grid", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = verify_patches(args.patches_dir, args.save_grid, args.preview_dir)
    failed_products = [p for p, s in results.items() if s["fail"] > 0]
    if failed_products:
        logger.warning(f"Products with failures: {failed_products}")
        sys.exit(1)
