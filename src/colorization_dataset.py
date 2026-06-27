"""
src/colorization_dataset.py  --  Phase 3 Dataset
Loads TIR@100m (input) and RGB@100m (target) patch pairs.
Computes LST-prior channel from normalised TIR values.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Scenes held out for validation (match Phase 2 split)
DEFAULT_VAL_SCENES = ["SCENE_009", "SCENE_010"]


class ColorizationDataset(Dataset):
    """
    Loads (tir_100m_512.npy, rgb_100m_512.npy) pairs from output/patches/.

    Generator input  : 2-channel tensor [TIR_norm, LST_prior]  float32 in [0,1]
    Generator target : 3-channel tensor [R, G, B]               float32 in [-1,1]
                       (scaled to [-1,1] to match Tanh output)
    """

    def __init__(self, patches_dir, scenes=None, augment=True):
        """
        Args:
            patches_dir : path to output/patches/
            scenes      : list of scene IDs to include; None = all scenes
            augment     : whether to apply random flip/rotate augmentation
        """
        self.augment = augment
        self.samples = []   # list of (tir_path, rgb_path)

        all_scenes = sorted(os.listdir(patches_dir))
        for scene in all_scenes:
            # Skip scenes not in the requested list
            if scenes is not None and scene not in scenes:
                continue
            scene_dir = os.path.join(patches_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            for sample in sorted(os.listdir(scene_dir)):
                sample_dir = os.path.join(scene_dir, sample)
                tir_path = os.path.join(sample_dir, "tir_100m_512.npy")
                rgb_path = os.path.join(sample_dir, "rgb_100m_512.npy")
                if os.path.exists(tir_path) and os.path.exists(rgb_path):
                    self.samples.append((tir_path, rgb_path, scene, sample))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _percentile_norm(arr, lo_pct=1, hi_pct=99):
        """Clip to [lo_pct, hi_pct] percentile and scale to [0, 1]."""
        lo = np.percentile(arr, lo_pct)
        hi = np.percentile(arr, hi_pct)
        if hi > lo:
            return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        return np.zeros_like(arr, dtype=np.float32)

    @staticmethod
    def _make_lst_prior(tir_norm):
        """
        Physics-informed LST prior channel.
        Normalized TIR (0-1) already correlates with Land Surface Temperature.
        We emphasise the hot/cold contrast with a slight sigmoid stretch.
        hot pixels (>0.6) -> urban/bare soil -> warm colors
        cold pixels (<0.3) -> water/snow   -> cool colors
        """
        return (1.0 / (1.0 + np.exp(-8.0 * (tir_norm - 0.5)))).astype(np.float32)

    def __getitem__(self, idx):
        tir_path, rgb_path, scene, sample = self.samples[idx]

        # Load raw arrays  (C, H, W) or (H, W)
        tir_raw = np.load(tir_path).astype(np.float32)
        rgb_raw = np.load(rgb_path).astype(np.float32)

        # Ensure channel-first  (1, H, W) and (3, H, W)
        if tir_raw.ndim == 2:
            tir_raw = tir_raw[np.newaxis]
        if rgb_raw.ndim == 2:
            rgb_raw = rgb_raw[np.newaxis]

        # Normalise TIR to [0, 1]
        tir_norm = self._percentile_norm(tir_raw)           # (1, 512, 512)

        # Normalise RGB per channel to [0, 1], then scale to [-1, 1]
        rgb_norm = np.zeros_like(rgb_raw, dtype=np.float32)
        for c in range(rgb_raw.shape[0]):
            rgb_norm[c] = self._percentile_norm(rgb_raw[c])
        rgb_scaled = rgb_norm * 2.0 - 1.0                   # [-1, 1] for Tanh

        # Compute physics-informed LST prior
        lst_prior = self._make_lst_prior(tir_norm)           # (1, 512, 512)

        # Stack into 2-channel generator input
        gen_input = np.concatenate([tir_norm, lst_prior], axis=0)  # (2, 512, 512)

        # Data augmentation (same transform applied to input and target)
        if self.augment:
            if random.random() > 0.5:
                gen_input = np.flip(gen_input, axis=-1).copy()
                rgb_scaled = np.flip(rgb_scaled, axis=-1).copy()
            if random.random() > 0.5:
                gen_input = np.flip(gen_input, axis=-2).copy()
                rgb_scaled = np.flip(rgb_scaled, axis=-2).copy()
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                gen_input = np.rot90(gen_input, k, axes=(-2, -1)).copy()
                rgb_scaled = np.rot90(rgb_scaled, k, axes=(-2, -1)).copy()

        return (
            torch.from_numpy(gen_input),     # (2, 512, 512) float32
            torch.from_numpy(rgb_scaled),    # (3, 512, 512) float32  in [-1,1]
            f"{scene}/{sample}",             # scene id for logging
        )


def build_color_dataloaders(patches_dir,
                             val_scenes=None,
                             batch_size=1,
                             num_workers=2):
    """Build train and validation DataLoaders for Phase 3."""
    if val_scenes is None:
        val_scenes = DEFAULT_VAL_SCENES

    all_scenes = [d for d in os.listdir(patches_dir)
                  if os.path.isdir(os.path.join(patches_dir, d))]
    train_scenes = [s for s in all_scenes if s not in val_scenes]

    train_ds = ColorizationDataset(patches_dir, scenes=train_scenes, augment=True)
    val_ds   = ColorizationDataset(patches_dir, scenes=val_scenes,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader


if __name__ == "__main__":
    import sys
    patches_dir = sys.argv[1] if len(sys.argv) > 1 else "output/patches"
    train_loader, val_loader = build_color_dataloaders(patches_dir, batch_size=1)
    print(f"Train samples : {len(train_loader.dataset)}")
    print(f"Val   samples : {len(val_loader.dataset)}")
    gen_in, rgb_tgt, sid = next(iter(train_loader))
    print(f"gen_input shape : {tuple(gen_in.shape)}  range [{gen_in.min():.3f}, {gen_in.max():.3f}]")
    print(f"rgb_target shape: {tuple(rgb_tgt.shape)}  range [{rgb_tgt.min():.3f}, {rgb_tgt.max():.3f}]")
    print(f"Scene/sample    : {sid[0]}")
    print("Dataset OK")
