"""
src/models/rrdbnet.py
─────────────────────────────────────────────────────────────────────────────
RRDBNet (Residual-in-Residual Dense Block Network) for ×2 Super-Resolution.

Architecture source: "ESRGAN: Enhanced Super-Resolution Generative Adversarial
Networks" (Wang et al., 2018) — the generator backbone also used by Real-ESRGAN.

Why RRDBNet for TIR Super-Resolution?
- Dense connections within each RRDB block allow gradient flow through many
  skip connections, preventing vanishing gradients during training.
- Residual scaling (β=0.2) stabilises training without batch normalization,
  which is important because TIR patches have highly variable statistics.
- 1-channel I/O: Unlike RGB ESRGAN, we use 1-ch input and 1-ch output since
  our TIR data is single-band (Band 10, thermal radiance).

Key architectural parameters (hackathon-tuned for speed vs quality):
  num_in_ch    = 1   (TIR single channel)
  num_out_ch   = 1   (SR output single channel)
  num_feat     = 64  (feature maps per layer)
  num_block    = 6   (number of RRDB blocks — reduced from 23 for speed)
  num_grow_ch  = 32  (dense block growth channels)
  scale        = 2   (256→512 upscaling)
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Leaky ReLU slope used throughout (same as original ESRGAN) ────────────
LRELU_SLOPE = 0.2


class DenseBlock(nn.Module):
    """
    A Dense Block with 5 conv layers.

    Each layer takes as input the concatenation of ALL previous feature maps
    (dense connectivity). This maximises feature reuse and gradient flow.

    Input/output channels:
        in_ch   → (in_ch + 4 * gc) → out_ch (via a final 1×1 projection)
    where gc = num_grow_ch (the growth rate per dense layer).
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat,              num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch,   num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat,    3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=LRELU_SLOPE, inplace=True)

        # Residual scaling factor (stabilises training)
        self.res_scale = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        # Final conv: no activation (raw residual)
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * self.res_scale + x   # residual connection


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB).

    Stacks 3 DenseBlocks with a global residual bypass.
    The 3-level residual structure (dense → residual → residual-in-residual)
    is why RRDBNet can be trained stably without batch normalisation.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = DenseBlock(num_feat, num_grow_ch)
        self.rdb2 = DenseBlock(num_feat, num_grow_ch)
        self.rdb3 = DenseBlock(num_feat, num_grow_ch)
        self.res_scale = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.res_scale + x  # global residual


class RRDBNet(nn.Module):
    """
    Full RRDBNet Generator for ×2 Super-Resolution (1-channel TIR).

    Architecture:
        Conv_in → [RRDB × num_block] → Conv_trunk → Upsample(×2) → Conv_HR → Conv_out

    The upsampling is done using nearest-neighbour interpolation followed by a
    convolution (PixelShuffle is an alternative, but this approach avoids
    checkerboard artefacts common in thermal images).

    Args:
        num_in_ch  : input channels  (1 for single-band TIR)
        num_out_ch : output channels (1 for single-band TIR)
        num_feat   : base feature channels (default 64)
        num_block  : number of RRDB blocks (default 6 for hackathon speed)
        num_grow_ch: growth channels in each dense block (default 32)
        scale      : upscaling factor (2 for 256→512)
    """

    def __init__(
        self,
        num_in_ch:  int = 1,
        num_out_ch: int = 1,
        num_feat:   int = 64,
        num_block:  int = 6,
        num_grow_ch: int = 32,
        scale:      int = 2,
    ):
        super().__init__()
        self.scale = scale

        # Initial feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # Body: stack of RRDB blocks
        self.body = nn.Sequential(*[
            RRDB(num_feat, num_grow_ch) for _ in range(num_block)
        ])

        # After-body trunk convolution
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling head
        # For ×2: one upsampling stage
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 4:
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # High-resolution refinement
        self.conv_hr  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=LRELU_SLOPE, inplace=True)

        # Weight initialisation (important for training stability)
        self._init_weights()

    def _init_weights(self):
        """Kaiming normal init for conv layers — standard for ESRGAN variants."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=LRELU_SLOPE, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input : (B, 1, 256, 256)
        Output: (B, 1, 512, 512)
        """
        feat = self.conv_first(x)

        # Body (RRDB blocks) + global residual skip
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat   # long-range residual

        # Upsampling ×2 via nearest-neighbour + conv (avoids checkerboard)
        feat = self.lrelu(
            self.conv_up1(
                F.interpolate(feat, scale_factor=2, mode='nearest')
            )
        )
        if self.scale == 4:
            feat = self.lrelu(
                self.conv_up2(
                    F.interpolate(feat, scale_factor=2, mode='nearest')
                )
            )

        # Final refinement
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def build_model(
    num_block: int = 6,
    num_feat:  int = 64,
    scale:     int = 2,
    device:    str = "cpu",
) -> RRDBNet:
    """
    Factory function to build and return an RRDBNet for TIR SR.

    Args:
        num_block : RRDB blocks. 6 = fast (hackathon), 23 = full Real-ESRGAN.
        num_feat  : Feature channels. 64 is standard.
        scale     : Upscaling factor (2 for our dataset).
        device    : 'cuda' or 'cpu'.

    Returns:
        model : RRDBNet instance on the specified device.
    """
    model = RRDBNet(
        num_in_ch=1,
        num_out_ch=1,
        num_feat=num_feat,
        num_block=num_block,
        num_grow_ch=num_feat // 2,
        scale=scale,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[RRDBNet] Parameters: {n_params:,}  |  Scale: x{scale}  |  Blocks: {num_block}")
    return model
