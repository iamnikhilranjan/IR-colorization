"""
src/dataset.py
─────────────────────────────────────────────────────────────────────────────
PyTorch Dataset for Thermal Infrared (TIR) Super-Resolution.

Reads paired .npy patches from output/patches/:
  ├── SCENE_XXX/
  │   └── sample_000/
  │       ├── tir_200m.npy      ← Low-Resolution input  (256×256)
  │       └── tir_100m_512.npy  ← High-Resolution target (512×512)

Train/Val split is done at the SCENE level (scene IDs) to prevent
spatial data leakage between train and validation sets.

Usage:
    from src.dataset import TIRPatchDataset, build_dataloaders
    train_loader, val_loader = build_dataloaders('output/patches', val_scenes=['SCENE_009','SCENE_010'])
─────────────────────────────────────────────────────────────────────────────
"""

import os
import glob
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ── Normalization statistics computed from thermal band physics ────────────
# Landsat 9 Band 10 (TIRS1): typical radiance range in W/(m²·sr·μm)
# We use per-sample percentile clipping + normalization for robustness.
CLIP_PERCENTILE_LO = 1.0
CLIP_PERCENTILE_HI = 99.0


def normalize_tir(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize a TIR array to [0, 1] using percentile clipping.

    Why percentile clipping?
    - Landsat 9 TIR data can have outlier pixels (cloud edges, sensor noise).
    - Clipping at 1st / 99th percentile removes these without distorting the
      thermal gradient information that the SR model needs to learn.

    Returns:
        normed  : float32 array in [0, 1]
        lo      : the clipping lower bound (for de-normalization)
        hi      : the clipping upper bound (for de-normalization)
    """
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, CLIP_PERCENTILE_LO))
    hi = float(np.percentile(arr, CLIP_PERCENTILE_HI))
    if hi == lo:  # constant patch (e.g., nodata region) — return zeros
        return np.zeros_like(arr), lo, hi
    normed = np.clip(arr, lo, hi)
    normed = (normed - lo) / (hi - lo + 1e-8)
    return normed, lo, hi


class TIRPatchDataset(Dataset):
    """
    PyTorch Dataset for paired TIR (LR, HR) patches.

    Each item returns:
        lr  : torch.Tensor  shape (1, 256, 256)  – TIR@200m (input to SR model)
        hr  : torch.Tensor  shape (1, 512, 512)  – TIR@100m (SR target)

    Args:
        patches_root : Path to the output/patches/ directory.
        scene_ids    : List of scene folder names to include (e.g. ['SCENE_001']).
                       If None, all scene directories are used.
        augment      : If True, apply random horizontal/vertical flips for training.
    """

    def __init__(
        self,
        patches_root: str,
        scene_ids: Optional[List[str]] = None,
        augment: bool = False,
    ):
        self.augment = augment
        self.samples: List[str] = []  # list of sample_dir paths

        patches_root = os.path.abspath(patches_root)
        if not os.path.isdir(patches_root):
            raise FileNotFoundError(f"patches_root not found: {patches_root}")

        # Discover scenes
        all_scenes = sorted(
            d for d in os.listdir(patches_root)
            if os.path.isdir(os.path.join(patches_root, d)) and d != "demo"
        )
        if scene_ids is not None:
            selected = [s for s in all_scenes if s in scene_ids]
        else:
            selected = all_scenes

        if not selected:
            raise ValueError(
                f"No matching scene directories found in '{patches_root}'. "
                f"Available: {all_scenes}"
            )

        # Collect all sample directories
        for scene in selected:
            scene_dir = os.path.join(patches_root, scene)
            for sample_dir in sorted(glob.glob(os.path.join(scene_dir, "sample_*"))):
                lr_path = os.path.join(sample_dir, "tir_200m.npy")
                hr_path = os.path.join(sample_dir, "tir_100m_512.npy")
                if os.path.isfile(lr_path) and os.path.isfile(hr_path):
                    self.samples.append(sample_dir)

        logger.info(
            f"TIRPatchDataset: {len(selected)} scenes, {len(self.samples)} samples, "
            f"augment={augment}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_dir = self.samples[idx]
        lr_np = np.load(os.path.join(sample_dir, "tir_200m.npy")).astype(np.float32)
        hr_np = np.load(os.path.join(sample_dir, "tir_100m_512.npy")).astype(np.float32)

        # Normalize independently (same scene thermal range but different resolution)
        lr_norm, _, _ = normalize_tir(lr_np)
        hr_norm, _, _ = normalize_tir(hr_np)

        # Ensure 2-D then add channel dim -> (1, H, W)
        if lr_norm.ndim == 2:
            lr_norm = lr_norm[np.newaxis, ...]   # (1, 256, 256)
        elif lr_norm.ndim == 3:
            lr_norm = lr_norm[:1]                # keep first channel only

        if hr_norm.ndim == 2:
            hr_norm = hr_norm[np.newaxis, ...]   # (1, 512, 512)
        elif hr_norm.ndim == 3:
            hr_norm = hr_norm[:1]

        lr_t = torch.from_numpy(lr_norm)
        hr_t = torch.from_numpy(hr_norm)

        # Data augmentation (only for training)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                lr_t = torch.flip(lr_t, dims=[-1])  # horizontal flip
                hr_t = torch.flip(hr_t, dims=[-1])
            if torch.rand(1).item() > 0.5:
                lr_t = torch.flip(lr_t, dims=[-2])  # vertical flip
                hr_t = torch.flip(hr_t, dims=[-2])
            if torch.rand(1).item() > 0.5:
                # 90-degree rotation
                lr_t = torch.rot90(lr_t, k=1, dims=[-2, -1])
                hr_t = torch.rot90(hr_t, k=1, dims=[-2, -1])

        return lr_t.contiguous(), hr_t.contiguous()


def build_dataloaders(
    patches_root: str,
    val_scenes: Optional[List[str]] = None,
    batch_size: int = 4,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders with an 8/2 scene split.

    Args:
        patches_root : Path to output/patches/
        val_scenes   : List of scene IDs to use for validation.
                       Defaults to ['SCENE_009', 'SCENE_010'].
        batch_size   : Batch size for training (default 4).
        num_workers  : Number of parallel data loading workers.

    Returns:
        train_loader, val_loader
    """
    if val_scenes is None:
        val_scenes = ["SCENE_009", "SCENE_010"]

    # Discover all available scenes (excluding demo)
    patches_root = os.path.abspath(patches_root)
    all_scenes = sorted(
        d for d in os.listdir(patches_root)
        if os.path.isdir(os.path.join(patches_root, d)) and d != "demo"
    )
    train_scenes = [s for s in all_scenes if s not in val_scenes]

    logger.info(f"Train scenes ({len(train_scenes)}): {train_scenes}")
    logger.info(f"Val   scenes ({len(val_scenes)}): {val_scenes}")

    train_ds = TIRPatchDataset(patches_root, scene_ids=train_scenes, augment=True)
    val_ds   = TIRPatchDataset(patches_root, scene_ids=val_scenes,   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"DataLoaders ready — train: {len(train_ds)} samples, "
        f"val: {len(val_ds)} samples"
    )
    return train_loader, val_loader
