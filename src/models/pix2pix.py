"""
src/models/pix2pix.py  --  Phase 3: Pix2Pix cGAN for TIR -> RGB Colorization
Generator  : U-Net (8-level encoder-decoder with skip connections)
             Input : (B, 2, 512, 512)  TIR@100m + LST-prior
             Output: (B, 3, 512, 512)  Predicted RGB in [-1, 1]
Discriminator : PatchGAN (70x70 receptive field)
             Input : (B, 4, 512, 512)  TIR (1ch) + RGB (3ch)
             Output: (B, 1, H', W')    Per-patch real/fake logits
"""

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """Conv2d stride-2 -> [InstanceNorm] -> LeakyReLU. Halves spatial dims."""
    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch, affine=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """ConvTranspose2d stride-2 -> InstanceNorm -> ReLU -> [Dropout]. Doubles spatial dims."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=False),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net generator for 512x512 images (ngf=64).
    Encoder  : 512->256->128->64->32->16->8->4->2  (8 DownBlocks)
    Decoder  : 2->4->8->16->32->64->128->256->512  (7 UpBlocks + final ConvT)
    in_channels=2  (TIR + LST-prior), out_channels=3 (RGB, Tanh output)
    """
    def __init__(self, in_channels=2, out_channels=3, ngf=64):
        super().__init__()
        # Encoder
        self.e1 = DownBlock(in_channels, ngf,    normalize=False)  # (64,  256,256)
        self.e2 = DownBlock(ngf,         ngf*2)                    # (128, 128,128)
        self.e3 = DownBlock(ngf*2,       ngf*4)                    # (256, 64, 64 )
        self.e4 = DownBlock(ngf*4,       ngf*8)                    # (512, 32, 32 )
        self.e5 = DownBlock(ngf*8,       ngf*8)                    # (512, 16, 16 )
        self.e6 = DownBlock(ngf*8,       ngf*8)                    # (512, 8,  8  )
        self.e7 = DownBlock(ngf*8,       ngf*8)                    # (512, 4,  4  )
        self.e8 = DownBlock(ngf*8,       ngf*8, normalize=False)   # (512, 2,  2  ) bottleneck

        # Decoder  (in_ch = prev_ch + skip_ch)
        self.d1 = UpBlock(ngf*8,     ngf*8, dropout=0.5)   # e8          -> (512,4,4)
        self.d2 = UpBlock(ngf*8*2,   ngf*8, dropout=0.5)   # d1+e7=1024  -> (512,8,8)
        self.d3 = UpBlock(ngf*8*2,   ngf*8, dropout=0.5)   # d2+e6=1024  -> (512,16,16)
        self.d4 = UpBlock(ngf*8*2,   ngf*8)                 # d3+e5=1024  -> (512,32,32)
        self.d5 = UpBlock(ngf*8*2,   ngf*4)                 # d4+e4=1024  -> (256,64,64)
        self.d6 = UpBlock(ngf*4*2,   ngf*2)                 # d5+e3=512   -> (128,128,128)
        self.d7 = UpBlock(ngf*2*2,   ngf)                   # d6+e2=256   -> (64,256,256)

        # Final: d7+e1 = 128ch -> 3ch RGB
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.e1(x);  e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        e5 = self.e5(e4); e6 = self.e6(e5); e7 = self.e7(e6); e8 = self.e8(e7)
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        return self.final(torch.cat([d7, e1], 1))


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator - 70x70 patch classification.
    Input : TIR (1ch) cat RGB (3ch) = 4ch
    Output: (B, 1, H', W') real/fake logits (use BCEWithLogitsLoss)
    """
    def __init__(self, in_channels=4, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf,   4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf,    ndf*2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2,  ndf*4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4,  ndf*8, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8,  1,     4, stride=1, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tir, rgb):
        return self.model(torch.cat([tir, rgb], 1))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = UNetGenerator(2, 3, 64).to(device)
    D = PatchGANDiscriminator(4, 64).to(device)
    x  = torch.randn(1, 2, 512, 512).to(device)
    out = G(x)
    disc = D(x[:, :1], out)
    gp = sum(p.numel() for p in G.parameters()) / 1e6
    dp = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"G output : {tuple(out.shape)}  (expected (1,3,512,512))")
    print(f"D output : {tuple(disc.shape)}")
    print(f"G params : {gp:.2f}M  |  D params : {dp:.2f}M")
    print(f"No NaN   : {not torch.isnan(out).any()}")
    print("PASSED")
