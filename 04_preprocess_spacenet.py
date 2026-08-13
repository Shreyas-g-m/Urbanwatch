from pathlib import Path
import json
import random
import shutil

import numpy as np
import tifffile
from PIL import Image
from shapely.geometry import shape


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

SPACENET_ROOT = ROOT / "AOI_3_Paris_Train"

IMAGE_DIR = SPACENET_ROOT / "RGB-PanSharpen"
GEOJSON_DIR = SPACENET_ROOT / "geojson" / "buildings"

OUTPUT_ROOT = ROOT / "data" / "yolo_buildings"

TRAIN_IMAGE_DIR = OUTPUT_ROOT / "images" / "train"
VAL_IMAGE_DIR = OUTPUT_ROOT / "images" / "val"

TRAIN_LABEL_DIR = OUTPUT_ROOT / "labels" / "train"
VAL_LABEL_DIR = OUTPUT_ROOT / "labels" / "val"

RANDOM_SEED = 42
VAL_RATIO = 0.20

# SpaceNet RGB-PanSharpen imagery is uint16.
# This clips reflectance values to a practical range before
# converting to 8-bit RGB.
UINT16_MAX = 2047


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    TRAIN_IMAGE_DIR,
    VAL_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    VAL_LABEL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# GEOREFERENCING
# ============================================================

def get_geotransform(tif_path):
    """
    Extract a simple north-up GeoTIFF transform using
    ModelPixelScaleTag and ModelTiepointTag.

    Returns:
        origin_x
        origin_y
        pixel_width
        pixel_height
    """

    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]

        tags = page.tags

        if "ModelPixelScaleTag" not in tags:
            raise RuntimeError(
                f"Missing ModelPixelScaleTag in {tif_path.name}"
            )

        if "ModelTiepointTag" not in tags:
            raise RuntimeError(
                f"Missing ModelTiepointTag in {tif_path.name}"
            )

        scale = tags["ModelPixelScaleTag"].value
        tiepoint = tags["ModelTiepointTag"].value

    pixel_width = float(scale[0])
    pixel_height = float(scale[1])

    origin_x = float(tiepoint[3])
    origin_y = float(tiepoint[4])

    return origin_x, origin_y, pixel_width, pixel_height


def geo_to_pixel(
    lon,
    lat,
    origin_x,
    origin_y,
    pixel_width,
    pixel_height,
):
    """
    Convert geographic coordinates to image pixel coordinates.

    SpaceNet GeoJSON is CRS84:
        X = longitude
        Y = latitude

    GeoTIFF uses:
        origin = upper-left
    """

    x = (lon - origin_x) / pixel_width

    y = (origin_y - lat) / pixel_height

    return x, y


# ============================================================
# TIFF → 8-BIT RGB
# ============================================================

def convert_tiff_to_rgb(tif_path):
    """
    Read a SpaceNet RGB-PanSharpen TIFF and convert uint16
    imagery to uint8 RGB.

    Returns:
        uint8 numpy array with shape H x W x 3
    """

    image = tifffile.imread(tif_path)

    if image.ndim != 3:
        raise ValueError(
            f"Unexpected TIFF shape {image.shape} for {tif_path.name}"
        )

    # Handle channel-first format if encountered
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] != 3:
        raise ValueError(
            f"Expected 3 channels, got {image.shape}"
        )

    image = image.astype(np.float32)

    image = np.clip(
        image,
        0,
        UINT16_MAX
    )

    image = (
        image / UINT16_MAX * 255.0
    ).astype(np.uint8)

    return image


# ============================================================
# GEOJSON → YOLO BOUNDING BOXES
# ============================================================

def geojson_to_yolo(
    geojson_path,
    image_width,
    image_height,
    transform,
):
    """
    Convert building polygons from GeoJSON into YOLO
    bounding-box annotations.

    YOLO format:

        class_id
        x_center
        y_center
        width
        height

    All coordinates normalized to [0, 1].
    """

    (
        origin_x,
        origin_y,
        pixel_width,
        pixel_height,
    ) = transform

    with open(
        geojson_path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    annotations = []

    for feature in data.get("features", []):

        geometry = feature.get("geometry")

        if geometry is None:
            continue

        try:
            polygon = shape(geometry)
        except Exception:
            continue

        if polygon.is_empty:
            continue

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty:
            continue

        min_lon, min_lat, max_lon, max_lat = polygon.bounds

        x1, y1 = geo_to_pixel(
            min_lon,
            max_lat,
            origin_x,
            origin_y,
            pixel_width,
            pixel_height,
        )

        x2, y2 = geo_to_pixel(
            max_lon,
            min_lat,
            origin_x,
            origin_y,
            pixel_width,
            pixel_height,
        )

        x_min = min(x1, x2)
        x_max = max(x1, x2)

        y_min = min(y1, y2)
        y_max = max(y1, y2)

        # Clip to image
        x_min = max(0, min(image_width, x_min))
        x_max = max(0, min(image_width, x_max))

        y_min = max(0, min(image_height, y_min))
        y_max = max(0, min(image_height, y_max))

        box_width = x_max - x_min
        box_height = y_max - y_min

        if box_width <= 1 or box_height <= 1:
            continue

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # Normalize
        x_center /= image_width
        y_center /= image_height
        box_width /= image_width
        box_height /= image_height

        annotations.append(
            f"0 {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    return annotations


# ============================================================
# FIND MATCHING GEOJSON
# ============================================================

def get_geojson_for_image(tif_path):

    image_id = tif_path.stem.replace(
        "RGB-PanSharpen_",
        ""
    )

    geojson_name = (
        f"buildings_{image_id}.geojson"
    )

    return GEOJSON_DIR / geojson_name


# ============================================================
# DATASET DISCOVERY
# ============================================================

tif_files = sorted(
    IMAGE_DIR.glob("*.tif")
)

print("=" * 70)
print("URBANWATCH — SPACENET PREPROCESSING")
print("=" * 70)

print("\nImages found:", len(tif_files))

if len(tif_files) == 0:
    raise RuntimeError(
        "No SpaceNet TIFF files were found."
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

random.seed(RANDOM_SEED)

files = tif_files.copy()

random.shuffle(files)

val_count = int(
    len(files) * VAL_RATIO
)

val_files = files[:val_count]
train_files = files[val_count:]

print("\nTrain images:", len(train_files))
print("Validation images:", len(val_files))


# ============================================================
# PROCESS DATASET
# ============================================================

statistics = {
    "train_images": 0,
    "val_images": 0,
    "train_buildings": 0,
    "val_buildings": 0,
    "empty_train_images": 0,
    "empty_val_images": 0,
    "failed": 0,
}


def process_split(
    files,
    image_output_dir,
    label_output_dir,
    split_name,
):

    print("\n" + "=" * 70)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 70)

    for index, tif_path in enumerate(files, start=1):

        try:

            # ------------------------------------------------
            # Read image
            # ------------------------------------------------

            image = convert_tiff_to_rgb(
                tif_path
            )

            height, width, channels = image.shape

            # ------------------------------------------------
            # GeoJSON
            # ------------------------------------------------

            geojson_path = get_geojson_for_image(
                tif_path
            )

            if not geojson_path.exists():

                print(
                    f"\nWARNING: Missing label for "
                    f"{tif_path.name}"
                )

                annotations = []

            else:

                transform = get_geotransform(
                    tif_path
                )

                annotations = geojson_to_yolo(
                    geojson_path,
                    width,
                    height,
                    transform,
                )

            # ------------------------------------------------
            # Output names
            # ------------------------------------------------

            output_name = (
                tif_path.stem + ".png"
            )

            image_output_path = (
                image_output_dir / output_name
            )

            label_output_path = (
                label_output_dir /
                (tif_path.stem + ".txt")
            )

            # ------------------------------------------------
            # Save image
            # ------------------------------------------------

            Image.fromarray(
                image
            ).save(
                image_output_path
            )

            # ------------------------------------------------
            # Save YOLO label
            # ------------------------------------------------

            with open(
                label_output_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    "\n".join(annotations)
                )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            if split_name == "train":

                statistics["train_images"] += 1
                statistics["train_buildings"] += len(
                    annotations
                )

                if len(annotations) == 0:
                    statistics["empty_train_images"] += 1

            else:

                statistics["val_images"] += 1
                statistics["val_buildings"] += len(
                    annotations
                )

                if len(annotations) == 0:
                    statistics["empty_val_images"] += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if index % 50 == 0 or index == 1:

                print(
                    f"[{index:4d}/{len(files)}] "
                    f"{tif_path.name} | "
                    f"buildings: {len(annotations)}"
                )

        except Exception as e:

            statistics["failed"] += 1

            print(
                f"\nERROR processing "
                f"{tif_path.name}: {e}"
            )


# ============================================================
# RUN
# ============================================================

process_split(
    train_files,
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    "train",
)

process_split(
    val_files,
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR,
    "val",
)


# ============================================================
# CREATE YOLO DATASET YAML
# ============================================================

yaml_content = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val

nc: 1
names:
  0: building
"""

yaml_path = OUTPUT_ROOT / "dataset.yaml"

with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(yaml_content)


# ============================================================
# SAVE PREPROCESSING SUMMARY
# ============================================================

summary_path = (
    OUTPUT_ROOT /
    "preprocessing_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        statistics,
        f,
        indent=4
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print("\nTrain images:",
      statistics["train_images"])

print("Validation images:",
      statistics["val_images"])

print("\nTrain building boxes:",
      statistics["train_buildings"])

print("Validation building boxes:",
      statistics["val_buildings"])

print("\nEmpty train images:",
      statistics["empty_train_images"])

print("Empty validation images:",
      statistics["empty_val_images"])

print("\nFailed images:",
      statistics["failed"])

print("\nOutput:")
print(OUTPUT_ROOT)

print("\nYOLO YAML:")
print(yaml_path)

print("\n" + "=" * 70)