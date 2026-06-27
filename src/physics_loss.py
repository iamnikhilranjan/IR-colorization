"""
src/physics_loss.py  --  Phase 4: Physics-Informed Loss (ISRO Bonus)

Encodes Stefan-Boltzmann thermal physics directly into GAN training:

  Rule 1: HOT pixels (TIR > 0.65) are urban/desert => should NOT be blue
          Penalty: mean(Blue_channel * hot_mask)

  Rule 2: COLD pixels (TIR < 0.35) are water/snow  => should NOT be red
          Penalty: mean(Red_channel * cold_mask)

  Rule 3: VERY HOT pixels (TIR > 0.80) => warm color dominance (R > B)
          Penalty: mean(clamp(B - R, min=0) * very_hot_mask)

LST Derivation (for reference, not computed here — we use normalized TIR):
  LST(Kelvin) = 0.00341802 * Band10_DN + 149.0
  Normalized TIR ~ proportional to LST after percentile clipping.
"""

import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """
    Temperature-to-color consistency loss.

    Args:
        hot_thresh    : TIR threshold above which pixel is "hot"  (default 0.65)
        cold_thresh   : TIR threshold below which pixel is "cold" (default 0.35)
        very_hot_thresh: TIR threshold for "very hot" (default 0.80)
    """

    def __init__(self, hot_thresh=0.65, cold_thresh=0.35, very_hot_thresh=0.80):
        super().__init__()
        self.hot_thresh      = hot_thresh
        self.cold_thresh     = cold_thresh
        self.very_hot_thresh = very_hot_thresh

    def forward(self, tir_norm, rgb_pred):
        """
        Args:
            tir_norm : (B, 1, H, W) normalized TIR in [0, 1]
            rgb_pred : (B, 3, H, W) predicted RGB in [-1, 1]  (Tanh output)

        Returns:
            physics_loss : scalar tensor
        """
        # Convert RGB from [-1,1] to [0,1]
        rgb = (rgb_pred + 1.0) / 2.0
        R = rgb[:, 0:1]   # (B, 1, H, W)
        G = rgb[:, 1:2]
        B = rgb[:, 2:3]

        eps = 1e-6

        # ── Rule 1: Hot pixels should not be blue ────────────────────────────
        hot_mask      = (tir_norm > self.hot_thresh).float()
        n_hot         = hot_mask.sum() + eps
        hot_blue      = (B * hot_mask).sum() / n_hot

        # ── Rule 2: Cold pixels should not be red ────────────────────────────
        cold_mask     = (tir_norm < self.cold_thresh).float()
        n_cold        = cold_mask.sum() + eps
        cold_red      = (R * cold_mask).sum() / n_cold

        # ── Rule 3: Very hot pixels must have warm dominance (R > B) ─────────
        very_hot_mask = (tir_norm > self.very_hot_thresh).float()
        n_very_hot    = very_hot_mask.sum() + eps
        # Penalise only when B > R (physically impossible for desert/urban)
        warm_violation = ((B - R).clamp(min=0) * very_hot_mask).sum() / n_very_hot

        return hot_blue + cold_red + warm_violation


def compute_physics_loss(tir_norm, rgb_pred,
                         hot_thresh=0.65, cold_thresh=0.35, very_hot_thresh=0.80):
    """Functional interface to PhysicsInformedLoss."""
    fn = PhysicsInformedLoss(hot_thresh, cold_thresh, very_hot_thresh)
    return fn(tir_norm, rgb_pred)


if __name__ == "__main__":
    loss_fn = PhysicsInformedLoss()

    # Test 1: Physically CORRECT prediction (hot=red, cold=blue) -> low loss
    tir  = torch.zeros(1, 1, 512, 512)
    tir[:, :, :256, :] = 0.9   # top half is HOT
    tir[:, :, 256:, :] = 0.1   # bottom half is COLD

    rgb_correct = torch.zeros(1, 3, 512, 512)
    rgb_correct[:, 0, :256, :] =  1.0  # hot  -> high R (→ mapped from [-1,1])
    rgb_correct[:, 2, 256:, :] =  1.0  # cold -> high B
    rgb_correct = rgb_correct * 2 - 1  # -> [-1,1]

    # Test 2: Physically WRONG prediction (hot=blue, cold=red) -> high loss
    rgb_wrong = torch.zeros(1, 3, 512, 512)
    rgb_wrong[:, 2, :256, :] = 1.0   # hot  -> high B  (WRONG)
    rgb_wrong[:, 0, 256:, :] = 1.0   # cold -> high R  (WRONG)
    rgb_wrong = rgb_wrong * 2 - 1

    loss_correct = loss_fn(tir, rgb_correct).item()
    loss_wrong   = loss_fn(tir, rgb_wrong).item()

    print(f"Physics loss (correct prediction) : {loss_correct:.4f}  (should be low)")
    print(f"Physics loss (wrong   prediction) : {loss_wrong:.4f}  (should be high)")
    assert loss_wrong > loss_correct, "Physics loss not discriminating!"
    print("PASSED")
