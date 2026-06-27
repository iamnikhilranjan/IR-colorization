"""
src/pipeline.py  --  Phase 4: End-to-End IR Colorization Pipeline

Chains Phase 2 (SR) + Phase 3 (Colorization) for single-command inference:
  TIR@200m (256x256) -> [RRDBNet x2] -> TIR@100m (512x512)
                     -> [LST prior]  -> 2ch input
                     -> [Pix2Pix G] -> RGB@100m (512x512)

Usage:
    from src.pipeline import IRColorizationPipeline
    pipe = IRColorizationPipeline("checkpoints/best_rrdbnet.pth",
                                  "checkpoints/best_pix2pix.pth")
    rgb, tir_sr = pipe(tir_lr_np)   # tir_lr_np: (1,256,256) numpy
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.rrdbnet import RRDBNet
from src.models.pix2pix import UNetGenerator


def _percentile_norm(arr, lo_pct=1, hi_pct=99):
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    if hi > lo:
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return np.zeros_like(arr, dtype=np.float32)


def _make_lst_prior(tir_norm):
    """Sigmoid-stretched LST prior channel from normalised TIR."""
    return (1.0 / (1.0 + np.exp(-8.0 * (tir_norm - 0.5)))).astype(np.float32)


class IRColorizationPipeline:
    """
    End-to-end TIR -> RGB colorization pipeline.

    Args:
        sr_checkpoint    : path to best_rrdbnet.pth
        color_checkpoint : path to best_pix2pix.pth  (None = skip colorization)
        device           : 'cpu', 'cuda', or 'auto'
        num_blocks       : RRDBNet RRDB blocks (default 6, must match training)
        ngf              : U-Net base filters (default 64, must match training)
    """

    def __init__(self, sr_checkpoint, color_checkpoint=None,
                 device="auto", num_blocks=6, ngf=64):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ── Load SR model ────────────────────────────────────────────────────
        self.sr = RRDBNet(in_channels=1, out_channels=1,
                          num_feat=64, num_block=num_blocks, scale=2).to(self.device)
        sr_ckpt = torch.load(sr_checkpoint, map_location=self.device, weights_only=False)
        self.sr.load_state_dict(sr_ckpt["model"])
        self.sr.eval()
        print(f"[Pipeline] SR model loaded from {sr_checkpoint}  "
              f"(epoch {sr_ckpt.get('epoch', '?')})")

        # ── Load Colorization model ───────────────────────────────────────────
        self.colorizer = None
        if color_checkpoint and os.path.exists(color_checkpoint):
            self.colorizer = UNetGenerator(
                in_channels=2, out_channels=3, ngf=ngf).to(self.device)
            col_ckpt = torch.load(color_checkpoint, map_location=self.device, weights_only=False)
            self.colorizer.load_state_dict(col_ckpt["G_state"])
            self.colorizer.eval()
            print(f"[Pipeline] Colorizer loaded from {color_checkpoint}  "
                  f"(epoch {col_ckpt.get('epoch', '?')})")
        else:
            print("[Pipeline] No colorizer checkpoint — SR-only mode.")

    @torch.no_grad()
    def run_sr(self, tir_lr_np):
        """
        Stage 1: Super-Resolution only.
        Args:
            tir_lr_np : (1, 256, 256) or (256, 256) numpy array (raw uint16 or float)
        Returns:
            tir_sr_np : (1, 512, 512) float32 numpy in [0, 1]
        """
        if tir_lr_np.ndim == 2:
            tir_lr_np = tir_lr_np[np.newaxis]

        tir_norm = _percentile_norm(tir_lr_np)                   # (1,256,256) [0,1]
        tir_t    = torch.from_numpy(tir_norm).unsqueeze(0).to(self.device)  # (1,1,256,256)
        tir_sr_t = self.sr(tir_t).clamp(0, 1)                   # (1,1,512,512)
        return tir_sr_t.squeeze(0).cpu().numpy()                  # (1,512,512)

    @torch.no_grad()
    def run_colorization(self, tir_sr_np):
        """
        Stage 2: Colorization only (requires SR output as input).
        Args:
            tir_sr_np : (1, 512, 512) float32 numpy in [0, 1]
        Returns:
            rgb_np    : (3, 512, 512) float32 numpy in [0, 1]
        """
        if self.colorizer is None:
            raise RuntimeError("No colorizer loaded. Pass color_checkpoint to __init__.")

        lst_prior = _make_lst_prior(tir_sr_np)                   # (1,512,512)
        gen_input = np.concatenate([tir_sr_np, lst_prior], axis=0)  # (2,512,512)
        gen_t     = torch.from_numpy(gen_input).unsqueeze(0).to(self.device)  # (1,2,512,512)
        rgb_t     = self.colorizer(gen_t)                         # (1,3,512,512) [-1,1]
        rgb_np    = ((rgb_t.squeeze(0).cpu().numpy() + 1) / 2).clip(0, 1)
        return rgb_np                                             # (3,512,512) [0,1]

    @torch.no_grad()
    def __call__(self, tir_lr_np):
        """
        Full end-to-end inference.
        Args:
            tir_lr_np : (1, 256, 256) or (256, 256) raw TIR@200m patch
        Returns:
            rgb_np    : (3, 512, 512) predicted RGB@100m in [0, 1]
            tir_sr_np : (1, 512, 512) SR TIR@100m in [0, 1]
        """
        tir_sr_np = self.run_sr(tir_lr_np)
        if self.colorizer is not None:
            rgb_np = self.run_colorization(tir_sr_np)
        else:
            # Grayscale fallback: repeat TIR 3 times
            rgb_np = np.repeat(tir_sr_np, 3, axis=0)
        return rgb_np, tir_sr_np


if __name__ == "__main__":
    # Sanity check: SR-only mode (no colorizer needed)
    pipe = IRColorizationPipeline(
        sr_checkpoint="checkpoints/best_rrdbnet.pth",
        color_checkpoint=None,
    )
    dummy = np.random.randint(0, 65535, (1, 256, 256), dtype=np.uint16).astype(np.float32)
    rgb, tir_sr = pipe(dummy)
    print(f"Input  TIR LR : {dummy.shape}")
    print(f"Output TIR SR : {tir_sr.shape}  range [{tir_sr.min():.3f}, {tir_sr.max():.3f}]")
    print(f"Output RGB    : {rgb.shape}    range [{rgb.min():.3f}, {rgb.max():.3f}]")
    print("Pipeline sanity check PASSED")
