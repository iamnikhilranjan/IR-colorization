This is a fantastic problem statement — let me break down everything you need: a clear solution architecture, the relevant research papers, and existing implementations to build on.

---

## Understanding the Problem (from ISRO's slides)

From the mentor session screenshots, the key clarifications are:

| Aspect | Detail |
|---|---|
| Input | TIR Band 10 (B10) @ **200m** resolution |
| Output 1 | Super-resolved TIR @ **100m** |
| Output 2 | Colorized RGB @ **100m** |
| Ground Truth RGB | Landsat 9 Bands B2+B3+B4 merged, downscaled to 100m |
| Official Repo | `github.com/jugal-sac/IR-colorization-BAH2026` |
| Bonus | Physics-informed modeling |

The workflow is: **Download B2,B3,B4,B10 → Patch extraction → SR model (200m→100m) → Colorization model → Evaluate**

---

## Recommended Solution Architecture

A two-stage sequential pipeline is the right approach.

### Stage 1 — Super-Resolution Module (TIR 200m → 100m, scale ×2)

The best proven approach for satellite thermal SR:

**Recommended model: SwinIR or Real-ESRGAN (RRDB-based)**

- **SwinIR** (Swin Transformer for Image Restoration) outperforms CNN-based methods on satellite imagery. For IR bands specifically, PSNR of ~24 and SSIM ~0.78 have been demonstrated, significantly outperforming all baselines.
- **ESRGAN/RRDB**: ESRGAN removes all batch norm layers from the residual block, replaces the basic block with RRDB (Residual-in-Residual Dense Block) which combines multi-layer residual networks and dense connections, and introduces a residual scaling factor β — resulting in finer details and fewer artifacts compared to standard SRGAN.
- For TIR specifically, attention mechanism integrated into the GAN framework has shown strong performance for infrared image super-resolution reconstruction.

**Loss function to use**: Perceptual (VGG) + L1 pixel loss + adversarial loss. No BN layers.

---

### Stage 2 — IR → RGB Colorization Module

This is the core challenge. Three strategies, ranked by expected performance:

**Option A (Best): cGAN with Pix2Pix + Semantic Constraint (Supervised)**
- TIC-CGAN proposes a conditional GAN for TIR→RGB colorization using a multi-term objective combining content, adversarial, perceptual, and total variation losses — each term contributing to both global color accuracy and fine local details.
- Pix2Pix trained on ~150K TIR-RGB pairs achieves an average SSIM of 0.58 for TIR-to-RGB translation, outperforming unpaired data methods.
- ISRO has confirmed they'll provide Landsat 9 TIR1-RGB **paired** data, so you can go fully supervised.

**Option B: CycleGAN / CUT (for data augmentation / unpaired scenarios)**
- CycleGAN-turbo introduces optimization techniques for faster learning and inference efficiency; CUT (Contrastive Unpaired Translation) performs contrastive learning to make corresponding patches learn similarly across domains.
- Use this only if paired coverage is insufficient.

**Option C (Bonus-worthy): Diffusion + ControlNet**
- Use a lightweight latent diffusion model conditioned on the TIR patch as ControlNet input for photorealistic colorization. Mentioned explicitly in ISRO's slides as a valid approach.

**For the Semantic Constraint (mandatory per PS)**:
- Run a pretrained land-cover classifier (e.g. a DeepLabV3+ or SegFormer fine-tuned on Landsat) on the TIR patches.
- Use class predictions to constrain color output: water → blue, vegetation → green, urban → grey-brown.
- Physics-Informed Hyperspectral Remote Sensing Image Synthesis with Deep conditional GANs has been demonstrated for IEEE TGRS — you can adapt this semantic constraint approach directly.

---

### Physics-Informed Bonus (High-impact differentiator!)

ISRO explicitly mentioned this as a **bonus** in the evaluation slide. Here's how to implement it:

1. **LST-based priors**: Landsat B10 can be converted to Land Surface Temperature (LST) using the formula `LST = m × B10 + b`. Cold pixels → water/vegetation (blue/green). Hot pixels → urban/bare soil (grey/brown/tan).
2. **Spectral indices from TIR**: Derive pseudo-NDVI and NDWI proxies using thermal contrast, then use these as auxiliary conditioning channels for the colorization network.
3. **Physics loss term**: Add a custom loss that penalizes the model whenever it assigns "water-like" colors (high blue) to thermally hot pixels. This encodes Stefan-Boltzmann radiation physics into training.

---

## Key Research Papers

Here are the most relevant papers with direct links:

| Paper | What it covers | Link |
|---|---|---|
| **TIC-CGAN** (Deng et al. 2020) | Core TIR→RGB colorization with cGAN, multi-term loss | [arxiv.org/pdf/1810.05399](https://arxiv.org/pdf/1810.05399) |
| **Pix2Pix** (Isola et al. 2017) | Foundation image-to-image translation framework | [arxiv.org/abs/1611.07004](https://arxiv.org/abs/1611.07004) |
| **CycleGAN** (Zhu et al. 2017) | Unpaired domain translation (useful backup) | [arxiv.org/abs/1703.10593](https://arxiv.org/abs/1703.10593) |
| **ESRGAN** (Wang et al. 2018) | RRDB-based super-resolution, state of the art | [arxiv.org/abs/1809.00219](https://arxiv.org/abs/1809.00219) |
| **SwinIR** (Liang et al. 2021) | Swin Transformer for image restoration/SR | [arxiv.org/abs/2108.10257](https://arxiv.org/abs/2108.10257) |
| **Infrared SR via GAN** (2023) | SR specifically for infrared images | [arxiv.org/pdf/2312.00689](https://arxiv.org/pdf/2312.00689) |
| **Enhancing TIR colorization** (2024) | GAN + contrastive learning for TIR detail fidelity | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1350449524005590) |
| **Nighttime TIR Colorization** (2023) | Feedback-based object appearance learning for TIR | [arxiv.org/pdf/2310.15688](https://arxiv.org/pdf/2310.15688) |
| **LSSR Landsat SR** (2025) | Multi-task diffusion for Landsat SR + IR | [arxiv.org/html/2510.23382v1](https://arxiv.org/html/2510.23382v1) |

---

## Existing Open-Source Implementations to Fork

These are directly usable starting points:

- **ISRO's official dataset prep**: `github.com/jugal-sac/IR-colorization-BAH2026` — start here first, it tells you exactly how to download and patch Landsat 9 data
- **BasicSR** (Pix2Pix/ESRGAN/SwinIR all in one): `github.com/XPixelGroup/BasicSR`
- **Real-ESRGAN** (production-ready SR): `github.com/xinntao/Real-ESRGAN`
- **Pix2Pix PyTorch**: `github.com/junyanz/pytorch-CycleGAN-and-pix2pix`

---

## End-to-End Implementation Plan (Hackathon Scope)

Given this is a hackathon with a prototype deliverable:

**Phase 1 — Data (Day 1 morning)**
Download 5–10 Landsat 9 scenes via USGS EarthExplorer or Google Earth Engine, follow ISRO's GitHub to create B10@200m, B10@100m, and RGB@100m paired patches of 256×256.

**Phase 2 — SR Model (Day 1 afternoon)**
Fine-tune a pretrained Real-ESRGAN checkpoint (×2 scale) on your TIR patches. This is fast because you're using transfer learning. Evaluate with PSNR/SSIM.

**Phase 3 — Colorization (Day 2)**
Train a Pix2Pix model (U-Net generator + PatchGAN discriminator) on (SR-TIR → RGB) pairs. Use combined L1 + adversarial + perceptual loss. Add a simple land-cover mask from a pretrained segmenter as an extra input channel.

**Phase 4 — Physics Bonus + Evaluation**
Add the LST-derived physics constraint as a weighted penalty in the colorization loss. Report PSNR, SSIM, FID on held-out scenes. Record inference time per 256×256 tile.

---

## Stack Summary

```
Data:       Landsat 9 (USGS EarthExplorer / GEE) → Rasterio + GDAL + tifffile
SR Module:  Real-ESRGAN / SwinIR (PyTorch, BasicSR framework)
Color Module: Pix2Pix cGAN (PyTorch) + semantic mask from DeepLabV3+
Physics:    LST from B10 → temperature-to-color prior loss term
Evaluation: piq library (PSNR, SSIM, FID), inference timing per tile
```

The physics-informed constraint is your biggest differentiator for scoring — none of the baseline implementations include it, and ISRO flagged it explicitly as a bonus. Focus on making that work even in a simple form (e.g., MSE penalty between predicted colors and a temperature-mapped palette).