"""
scripts/download_multi_scenes.py
─────────────────────────────────────────────────────────────────────────────
Download 5–10 Landsat 9 (LC09) scenes via Google Earth Engine.

Scenes are selected to cover diverse Indian terrain types so the training
dataset generalises well across land-cover categories.

Usage
─────
    # Authenticate once (opens browser):
    earthengine authenticate

    # Then download all scenes:
    python scripts/download_multi_scenes.py --project <your-gee-project-id>

    # Dry-run (print scenes, don't download):
    python scripts/download_multi_scenes.py --project <id> --dry_run

    # Single scene:
    python scripts/download_multi_scenes.py --project <id> --scene_ids SCENE_001

Output structure (ready for driver.py):
    input/
    └── <scene_id>/
        ├── <scene_id>_B2.TIF
        ├── <scene_id>_B3.TIF
        ├── <scene_id>_B4.TIF
        └── <scene_id>_B10.TIF
─────────────────────────────────────────────────────────────────────────────
"""

import ee
import geemap
import os
import sys
import argparse
import logging
import json
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Scene Catalogue ────────────────────────────────────────────────────────
# Each entry defines one Landsat 9 WRS-2 tile over a distinct terrain type.
# WRS Path/Row: https://landsat.usgs.gov/landsat_acq
# Cloud cover filter: <= CLOUD_COVER_MAX %
# Date range: scenes acquired in this window will be searched.

SCENE_CATALOGUE = [
    {
        "id": "SCENE_001",
        "description": "Rajasthan Desert — Thar (arid, sandy terrain)",
        "wrs_path": 148,
        "wrs_row": 41,
        "start_date": "2023-10-01",
        "end_date": "2024-03-31",
    },
    {
        "id": "SCENE_002",
        "description": "Delhi NCR — Dense urban heat island",
        "wrs_path": 146,
        "wrs_row": 40,
        "start_date": "2023-11-01",
        "end_date": "2024-02-28",
    },
    {
        "id": "SCENE_003",
        "description": "Kerala Coast — Tropical, coastal, high vegetation",
        "wrs_path": 144,
        "wrs_row": 54,
        "start_date": "2023-01-01",
        "end_date": "2023-04-30",
    },
    {
        "id": "SCENE_004",
        "description": "Gangetic Plains (UP/Bihar) — Agricultural flatlands",
        "wrs_path": 147,
        "wrs_row": 41,
        "start_date": "2023-11-01",
        "end_date": "2024-02-28",
    },
    {
        "id": "SCENE_005",
        "description": "Himalayan Foothills (Uttarakhand) — Snow + forest",
        "wrs_path": 147,
        "wrs_row": 38,
        "start_date": "2023-02-01",
        "end_date": "2023-05-31",
    },
    {
        "id": "SCENE_006",
        "description": "Mumbai Metropolitan — Coastal urban, water bodies",
        "wrs_path": 148,
        "wrs_row": 47,
        "start_date": "2023-11-01",
        "end_date": "2024-02-28",
    },
    {
        "id": "SCENE_007",
        "description": "Deccan Plateau (Telangana) — Semi-arid, mixed crops",
        "wrs_path": 144,
        "wrs_row": 47,
        "start_date": "2023-01-01",
        "end_date": "2023-04-30",
    },
    {
        "id": "SCENE_008",
        "description": "Sundarbans (WB) — Mangrove + tidal rivers",
        "wrs_path": 138,
        "wrs_row": 44,
        "start_date": "2023-11-01",
        "end_date": "2024-02-28",
    },
    {
        "id": "SCENE_009",
        "description": "Rann of Kutch — Salt flats + coastal wetlands",
        "wrs_path": 150,
        "wrs_row": 43,
        "start_date": "2023-11-01",
        "end_date": "2024-02-28",
    },
    {
        "id": "SCENE_010",
        "description": "Kaziranga / Brahmaputra (Assam) — Floodplains + grassland",
        "wrs_path": 136,
        "wrs_row": 42,
        "start_date": "2023-11-01",
        "end_date": "2024-02-28",
    },
]

# Bands to download (Landsat 9 Collection-2, Level-2 surface reflectance)
BANDS = ["SR_B2", "SR_B3", "SR_B4", "ST_B10"]

# Band-to-suffix mapping for output filenames expected by driver.py
BAND_SUFFIX_MAP = {
    "SR_B2": "_B2",
    "SR_B3": "_B3",
    "SR_B4": "_B4",
    "ST_B10": "_B10",
}

CLOUD_COVER_MAX = 10  # percent


# ── Core download logic ────────────────────────────────────────────────────

def init_gee(project_id: str):
    """Initialize the Earth Engine API."""
    try:
        ee.Initialize(project=project_id)
        logger.info(f"GEE initialized with project: {project_id}")
    except Exception as e:
        logger.error(
            f"GEE initialization failed: {e}\n"
            "Run 'earthengine authenticate' first, then retry."
        )
        sys.exit(1)


def get_best_image(scene: dict) -> ee.Image | None:
    """
    Fetch the least-cloudy Landsat 9 image for the given WRS path/row
    within the date range. Returns None if no image found.
    """
    collection = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterDate(scene["start_date"], scene["end_date"])
        .filterMetadata("WRS_PATH", "equals", scene["wrs_path"])
        .filterMetadata("WRS_ROW", "equals", scene["wrs_row"])
        .filterMetadata("CLOUD_COVER", "less_than", CLOUD_COVER_MAX)
        .select(BANDS)
        .sort("CLOUD_COVER")  # least cloudy first
    )

    count = collection.size().getInfo()
    if count == 0:
        logger.warning(
            f"[{scene['id']}] No image found (path={scene['wrs_path']}, "
            f"row={scene['wrs_row']}, {scene['start_date']}–{scene['end_date']}, "
            f"cloud<{CLOUD_COVER_MAX}%)"
        )
        return None

    image = collection.first()
    props = image.toDictionary(["LANDSAT_PRODUCT_ID", "DATE_ACQUIRED", "CLOUD_COVER"]).getInfo()
    logger.info(
        f"[{scene['id']}] Found: {props.get('LANDSAT_PRODUCT_ID', 'N/A')} | "
        f"Date: {props.get('DATE_ACQUIRED', 'N/A')} | "
        f"Cloud: {props.get('CLOUD_COVER', 'N/A'):.1f}%"
    )
    return image


def download_scene(scene: dict, output_root: str, scale: int = 30) -> bool:
    """
    Download all 4 bands (B2, B3, B4, B10) for a single scene to:
        output_root/<scene_id>/<scene_id>_<BAND>.TIF

    Returns True on success, False on failure.
    """
    output_dir = os.path.join(output_root, scene["id"])
    os.makedirs(output_dir, exist_ok=True)

    # Check if already downloaded
    existing = [
        f for f in os.listdir(output_dir)
        if f.lower().endswith(".tif")
    ]
    if len(existing) >= 4:
        logger.info(f"[{scene['id']}] Already downloaded ({len(existing)} TIFs found). Skipping.")
        return True

    image = get_best_image(scene)
    if image is None:
        return False

    # Get the centroid of the image and create a 30km buffer (60km x 60km bounding box)
    # This ensures the download size stays under the GEE 50MB limit for direct downloads.
    # 60km x 60km at 30m resolution is 2000x2000 pixels (~16MB per band).
    centroid = image.geometry().centroid()
    bounds = centroid.buffer(30000).bounds()

    logger.info(f"[{scene['id']}] Downloading 4 bands to: {output_dir}")

    for gee_band, suffix in BAND_SUFFIX_MAP.items():
        out_path = os.path.join(output_dir, f"{scene['id']}{suffix}.TIF")

        if os.path.exists(out_path):
            logger.info(f"  [skip] {os.path.basename(out_path)} already exists")
            continue

        logger.info(f"  Downloading {gee_band} → {os.path.basename(out_path)} ...")
        try:
            single_band = image.select([gee_band])
            geemap.ee_export_image(
                single_band,
                filename=out_path,
                scale=scale,
                region=bounds,
                file_per_band=False,
                crs="EPSG:4326",
            )
            logger.info(f"  ✓ {os.path.basename(out_path)}")
        except Exception as e:
            logger.error(f"  ✗ Failed to download {gee_band}: {e}")
            return False

    # Write metadata sidecar
    meta_path = os.path.join(output_dir, "scene_meta.json")
    image_props = image.toDictionary(
        ["LANDSAT_PRODUCT_ID", "DATE_ACQUIRED", "CLOUD_COVER",
         "WRS_PATH", "WRS_ROW", "SPACECRAFT_ID"]
    ).getInfo()
    meta = {**scene, "gee_properties": image_props}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[{scene['id']}] ✓ Complete — metadata saved to {meta_path}")
    return True


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download 5–10 diverse Landsat 9 scenes via Google Earth Engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Your Google Earth Engine Cloud project ID.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="input",
        help="Root directory for downloaded scenes (default: input/)",
    )
    parser.add_argument(
        "--scene_ids",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Specific scene IDs to download (e.g. SCENE_001 SCENE_003). "
            "Defaults to all 10 scenes."
        ),
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=30,
        help="Native pixel scale in metres for download (default: 30 = native Landsat resolution).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print scene catalogue and exit without downloading.",
    )
    parser.add_argument(
        "--cloud_max",
        type=int,
        default=CLOUD_COVER_MAX,
        help=f"Max cloud cover %% to accept (default: {CLOUD_COVER_MAX}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Dry run — just list scenes
    if args.dry_run:
        print("\n── Landsat 9 Scene Catalogue ────────────────────────────────")
        for s in SCENE_CATALOGUE:
            print(
                f"  {s['id']:12s}  Path={s['wrs_path']:3d}/Row={s['wrs_row']:3d}  "
                f"{s['start_date']} → {s['end_date']}  {s['description']}"
            )
        print()
        return

    # Filter catalogue
    scenes_to_download = SCENE_CATALOGUE
    if args.scene_ids:
        valid = {s["id"] for s in SCENE_CATALOGUE}
        for sid in args.scene_ids:
            if sid not in valid:
                logger.error(f"Unknown scene ID: {sid}. Valid: {sorted(valid)}")
                sys.exit(1)
        scenes_to_download = [s for s in SCENE_CATALOGUE if s["id"] in args.scene_ids]

    # Override cloud threshold if provided
    global CLOUD_COVER_MAX
    CLOUD_COVER_MAX = args.cloud_max

    # Initialize GEE
    init_gee(args.project)

    # Download
    logger.info(f"Starting download of {len(scenes_to_download)} scene(s) → {args.output_dir}/")
    success, failed = [], []

    for scene in scenes_to_download:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Scene {scene['id']}: {scene['description']}")
        ok = download_scene(scene, output_root=args.output_dir, scale=args.scale)
        (success if ok else failed).append(scene["id"])

    # Summary
    print(f"\n{'═'*60}")
    print(f"✓ Downloaded : {len(success)} scene(s): {', '.join(success) or 'none'}")
    print(f"✗ Failed     : {len(failed)} scene(s): {', '.join(failed) or 'none'}")
    print(f"\nNext step: python driver.py")
    print(f"{'═'*60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
