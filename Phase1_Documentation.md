# Phase 1: Data Acquisition & Processing — Complete Guide

Welcome to the documentation for **Phase 1** of our Thermal Infrared (TIR) Super-Resolution and Colorization project. This guide is written so that anyone—even a beginner—can understand exactly what we have built so far, why we built it, and how it works. 

This document is also designed to help the team answer questions from judges and mentors during the hackathon.

---

## 🌟 The Big Picture

### The Problem
Satellites like Landsat 9 capture images of the Earth in different "bands" (wavelengths of light). 
- **RGB bands** (Red, Green, Blue) are captured at a high resolution of **30 meters per pixel**.
- **TIR bands** (Thermal Infrared, which measures temperature/heat) are captured at a lower resolution of **100 meters per pixel**.

Because the thermal data is lower resolution, it looks blurry. Furthermore, thermal images are just grayscale (showing heat intensity), making them hard for human eyes to interpret compared to natural color (RGB) images.

### Our Goal for the Hackathon
1. **Super-Resolution (Phase 2):** Use AI to sharpen the blurry 100m Thermal images to a much clearer resolution.
2. **Colorization (Phase 3):** Train an AI to accurately "guess" the natural colors of a landscape just by looking at its thermal heat signature.

### What We Did in Phase 1
Before we can train any AI, we need **data**. We can't just feed the AI random images; it needs perfectly matched pairs of "Blurry Thermal", "Sharp Thermal", and "RGB". 

Phase 1 was entirely about building a pipeline to automatically download, align, and slice raw satellite data into perfect training examples (called "patches") for our AI.

---

## 🛠️ Step-by-Step Breakdown of What We Built

We created a pipeline that does the following steps automatically:

### Step 1: Downloading the Data (`scripts/download_multi_scenes.py`)
We use Google Earth Engine (GEE) to find and download raw satellite images. 
- We selected 10 different geographical scenes across India to ensure our AI learns diverse terrains (cities, forests, deserts, etc.).
- For each scene, we download the thermal band (Band 10) and the RGB bands (Bands 4, 3, 2).

### Step 2: Merging RGB Bands (`scripts/merge_rgb.py`)
Satellites download Red, Green, and Blue as separate grayscale files. This script takes those three files and stacks them on top of each other to create a single, standard color image at 30m resolution.

### Step 3: Simulating the "Blurry" Input (`scripts/downscale.py`)
To train an AI to sharpen an image, you must give it a blurry image and show it the sharp answer. 
- We take our thermal image and **intentionally blur it** to 200m resolution. This acts as our "Input".
- The original 100m thermal image acts as the "Ground Truth" (the correct answer the AI will try to guess).
- We also scale the RGB images to 100m to match the sharp thermal images for the colorization phase.

### Step 4: Slicing into Patches (`scripts/create_patches.py`)
Satellite images are massive (thousands of pixels wide). AI models can't process them all at once without running out of memory. 
- This script acts like a cookie-cutter. It slices the massive images into small squares (patches) of `256x256` pixels.
- **CRITICAL FIX WE MADE:** We encountered a bug where patches from different scenes were getting mixed together because of how the files were named. We fixed the script to properly extract the `SCENE_ID` so that every patch stays perfectly matched with its exact geographic location.

### Step 5: Verification (`scripts/verify_patches.py`)
If our "Blurry Input" square doesn't perfectly geographically overlap with our "Sharp Answer" square, the AI will learn garbage. This script acts as our quality control. It mathematically checks the alignment between the layers. **All 10 of our scenes successfully passed this test!**

---

## 🎤 Hackathon Q&A Cheat Sheet

Here are questions judges or mentors might ask, and how you as a team can answer them confidently:

### Q1: "Why did you build your own data pipeline instead of using a pre-existing dataset?"
**Answer:** "There are very few pre-aligned datasets specifically for Thermal Infrared Super-Resolution mapped to RGB. By building our own pipeline using Google Earth Engine, we have total control over the data. It allowed us to specifically target diverse Indian terrains (like urban areas, agriculture, and water bodies) which makes our AI highly relevant for local ISRO use cases."

### Q2: "How do you know your data is correctly aligned? If the RGB and Thermal don't match, your AI will fail."
**Answer:** "We built a dedicated verification script (`verify_patches.py`) that strictly checks the alignment. It calculates the Mean Absolute Error (MAE) between the spatial coordinates of the patches. If a patch is misaligned beyond a strict threshold, our pipeline throws an error and rejects it. We successfully passed 11/11 tests across all our downloaded scenes."

### Q3: "Why did you downscale the data to 200m instead of just using the raw data?"
**Answer:** "Because this is a Super-Resolution task using Transfer Learning. To teach the AI to upscale from 100m to 50m, we simulate the problem by downscaling the 100m to 200m. The model learns the mathematical relationship of 'upscaling by 2x'. Once trained, we can apply that exact same trained model to the real 100m data to upscale it to 50m."

### Q4: "What was the hardest technical challenge in Phase 1?"
**Answer:** "Handling the sheer scale of the TIF files and ensuring perfect coregistration (alignment). We initially had a bug where the patching script was overwriting output folders because it couldn't distinguish between different scenes due to an underscore in the naming convention. We had to fix the string parsing logic so that each scene was grouped perfectly, preventing the data from becoming a blended, misaligned mess."

### Q5: "What is the final format of the data you pass to the AI?"
**Answer:** "We save the data as raw `.npy` (NumPy arrays) because it preserves the exact scientific float values of the thermal radiance, which would be lost if we saved them as standard `.png` or `.jpg` images. We only generate `.pngs` alongside them for human visualization and debugging."

---

## Next Steps: Moving to Phase 2

With Phase 1 complete, our repository's `output/patches/` directory is now full of perfect, verified `.npy` squares. 

In **Phase 2**, we will build a PyTorch training loop and feed these squares into an **RRDBNet** (the architecture behind Real-ESRGAN). The model will look at the 200m squares, try to sharpen them, compare its guess against the 100m squares, and update its weights to get smarter over time!
