#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_and_run.sh
# Phase 1 — One-shot setup, authentication, download, and patch generation.
#
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh --project <your-gee-project-id>
#
# Prerequisites:
#   - Python 3.9+
#   - pip
#   - Internet connection (for GEE + pip packages)
#   - A Google Earth Engine account: https://signup.earthengine.google.com
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Load environment variables from .env if it exists ──────────────────────────
GEE_PROJECT=""
if [[ -f .env ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}" # Strip comments
        line=$(echo "$line" | xargs 2>/dev/null || echo "$line") # Trim whitespace
        if [[ "$line" =~ ^([^=]+)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            val="${BASH_REMATCH[2]}"
            key=$(echo "$key" | xargs 2>/dev/null || echo "$key")
            val=$(echo "$val" | xargs 2>/dev/null || echo "$val")
            if [[ "$key" == "GEE_PROJECT" ]]; then
                GEE_PROJECT="$val"
            fi
        fi
    done < .env
fi

# ── Parse args ────────────────────────────────────────────────────────────────
SKIP_INSTALL=false
DRY_RUN=false
SCENE_IDS=""   # optional: space-separated e.g. "SCENE_001 SCENE_002"
CLOUD_MAX=10

print_usage() {
    echo "Usage: $0 --project <gee-project-id> [options]"
    echo ""
    echo "Options:"
    echo "  --project    <id>     Google Earth Engine project ID (required)"
    echo "  --skip_install        Skip pip install step"
    echo "  --dry_run             List scenes, don't download"
    echo "  --scenes     <ids>    Quoted space-separated scene IDs (default: all 10)"
    echo "  --cloud_max  <int>    Max cloud cover % (default: 10)"
    echo ""
    echo "Examples:"
    echo "  $0 --project my-gee-proj"
    echo "  $0 --project my-gee-proj --scenes 'SCENE_001 SCENE_003 SCENE_005'"
    echo "  $0 --project my-gee-proj --dry_run"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)     GEE_PROJECT="$2"; shift 2 ;;
        --skip_install) SKIP_INSTALL=true; shift ;;
        --dry_run)     DRY_RUN=true; shift ;;
        --scenes)      SCENE_IDS="$2"; shift 2 ;;
        --cloud_max)   CLOUD_MAX="$2"; shift 2 ;;
        -h|--help)     print_usage; exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

if [[ -z "$GEE_PROJECT" && "$DRY_RUN" == "false" ]]; then
    echo "Error: --project is required."
    print_usage
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  IR Colorization BAH2026 — Phase 1: Data Preparation"
echo "══════════════════════════════════════════════════════════════════"
echo "  Project dir : $PROJECT_DIR"
echo "  GEE project : ${GEE_PROJECT:-<dry-run mode>}"
echo ""

# ── Step 1: Install dependencies ──────────────────────────────────────────────
if [[ "$SKIP_INSTALL" == "false" ]]; then
    echo "── [1/5] Installing Python dependencies ──────────────────────────"
    # Install core packages (without GDAL line which may need system lib)
    pip install --quiet --break-system-packages \
        "earthengine-api>=0.1.370" \
        "geemap>=0.30.0" \
        "rasterio>=1.3.0" \
        "tifffile>=2023.1.1" \
        "numpy>=1.24.0" \
        "opencv-python>=4.8.0" \
        "Pillow>=10.0.0"
    echo "  ✓ Dependencies installed"
else
    echo "── [1/5] Skipping dependency install (--skip_install)"
fi

# ── Step 2: GEE Authentication ────────────────────────────────────────────────
echo ""
echo "── [2/5] Google Earth Engine authentication ──────────────────────"
echo "  Checking GEE credentials..."

if python3 -c "import ee; ee.Initialize(project='${GEE_PROJECT:-test}')" 2>/dev/null; then
    echo "  ✓ GEE already authenticated"
else
    echo "  Running 'earthengine authenticate' — a browser window will open."
    echo "  Follow the steps, paste the token, then re-run this script."
    earthengine authenticate
    echo "  ✓ Authentication complete"
fi

# ── Step 3: Download scenes ───────────────────────────────────────────────────
echo ""
echo "── [3/5] Downloading Landsat 9 scenes ────────────────────────────"

DOWNLOAD_CMD="python3 scripts/download_multi_scenes.py --project $GEE_PROJECT --cloud_max $CLOUD_MAX"

if [[ -n "$SCENE_IDS" ]]; then
    DOWNLOAD_CMD="$DOWNLOAD_CMD --scene_ids $SCENE_IDS"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    DOWNLOAD_CMD="$DOWNLOAD_CMD --dry_run"
fi

echo "  Running: $DOWNLOAD_CMD"
eval "$DOWNLOAD_CMD"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "  Dry-run complete. Exiting."
    exit 0
fi

# ── Step 4: Run driver.py pipeline ───────────────────────────────────────────
echo ""
echo "── [4/5] Running dataset generation pipeline (driver.py) ─────────"
python3 driver.py
echo "  ✓ Patches generated in output/patches/"

# ── Step 5: Verify patches ────────────────────────────────────────────────────
echo ""
echo "── [5/5] Verifying patches ────────────────────────────────────────"
python3 scripts/verify_patches.py --patches_dir output/patches --save_grid
echo "  ✓ Preview grids saved to output/patch_previews/"

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Phase 1 Complete!"
echo "  Training patches : output/patches/"
echo "  Preview images   : output/patch_previews/"
echo "  Next step        : Phase 2 — SR model training"
echo "══════════════════════════════════════════════════════════════════"
echo ""
