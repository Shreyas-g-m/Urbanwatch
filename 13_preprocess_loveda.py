# ============================================================
# URBANWATCH — LoveDA PREPROCESSING
# 1024x1024 -> 4 x 512x512 crops
# ============================================================

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import json
import time


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

PROCESSED_ROOT = (
    ROOT
    / "data"
    / "loveda"
)

# Original LoveDA folders
SOURCE_ROOT = ROOT


# ============================================================
# CONFIGURATION
# ============================================================

CROP_SIZE = 512

REGIONS = [
    "Rural",
    "Urban"
]

SPLITS = {
    "train": "Train",
    "val": "Val"
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("URBANWATCH — LoveDA PREPROCESSING")
print("=" * 70)

print("Project root:")
print(ROOT)

print("\nOutput:")
print(PROCESSED_ROOT)

print("\nCrop size:")
print(f"{CROP_SIZE} x {CROP_SIZE}")


# ============================================================
# CHECK SOURCE DATA
# ============================================================

print("\n" + "=" * 70)
print("CHECKING SOURCE DATA")
print("=" * 70)

for source_split in SPLITS.values():

    split_path = (
        SOURCE_ROOT
        / source_split
    )

    if not split_path.exists():

        raise FileNotFoundError(
            f"LoveDA folder not found:\n{split_path}"
        )

    print(
        f"{source_split}: OK"
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in SPLITS:

    (
        PROCESSED_ROOT
        / split
        / "images"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        PROCESSED_ROOT
        / split
        / "masks"
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CROP FUNCTION
# ============================================================

def create_crops(image, mask):

    h, w = mask.shape

    if h != 1024 or w != 1024:

        raise ValueError(
            f"Expected 1024x1024 mask, "
            f"got {h}x{w}"
        )

    crops = []

    positions = [
        (0, 0),
        (0, 512),
        (512, 0),
        (512, 512)
    ]

    for top, left in positions:

        image_crop = image[
            top:top + CROP_SIZE,
            left:left + CROP_SIZE
        ]

        mask_crop = mask[
            top:top + CROP_SIZE,
            left:left + CROP_SIZE
        ]

        crops.append(
            (
                image_crop,
                mask_crop
            )
        )

    return crops


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    output_split,
    source_split
):

    print("\n" + "=" * 70)
    print(
        f"PROCESSING {output_split.upper()}"
    )
    print("=" * 70)

    source_root = (
        SOURCE_ROOT
        / source_split
    )

    output_image_dir = (
        PROCESSED_ROOT
        / output_split
        / "images"
    )

    output_mask_dir = (
        PROCESSED_ROOT
        / output_split
        / "masks"
    )

    total_source_images = 0
    total_pairs = 0
    total_crops = 0
    failed = 0

    start_time = time.time()

    for region in REGIONS:

        image_dir = (
            source_root
            / region
            / "images_png"
        )

        mask_dir = (
            source_root
            / region
            / "masks_png"
        )

        image_files = sorted(
            image_dir.glob("*.png")
        )

        print(
            f"\n{region}: "
            f"{len(image_files)} images"
        )

        region_pairs = 0
        region_crops = 0

        for index, image_path in enumerate(
            image_files,
            start=1
        ):

            total_source_images += 1

            mask_path = (
                mask_dir
                / image_path.name
            )

            # ------------------------------------------------
            # Missing mask
            # ------------------------------------------------

            if not mask_path.exists():

                print(
                    f"\nWARNING: Missing mask: "
                    f"{image_path.name}"
                )

                failed += 1

                continue

            try:

                # ------------------------------------------------
                # Load image
                # ------------------------------------------------

                image = np.array(
                    Image.open(
                        image_path
                    ).convert("RGB")
                )

                # ------------------------------------------------
                # Load mask
                # ------------------------------------------------

                mask = np.array(
                    Image.open(
                        mask_path
                    )
                )

                # ------------------------------------------------
                # Validate
                # ------------------------------------------------

                if image.shape != (
                    1024,
                    1024,
                    3
                ):

                    raise ValueError(
                        f"Invalid image shape: "
                        f"{image.shape}"
                    )

                if mask.shape != (
                    1024,
                    1024
                ):

                    raise ValueError(
                        f"Invalid mask shape: "
                        f"{mask.shape}"
                    )

                # ------------------------------------------------
                # Create four crops
                # ------------------------------------------------

                crops = create_crops(
                    image,
                    mask
                )

                for crop_index, (
                    image_crop,
                    mask_crop
                ) in enumerate(crops):

                    output_name = (
                        f"{region.lower()}_"
                        f"{image_path.stem}_"
                        f"{crop_index}.png"
                    )

                    image_output = (
                        output_image_dir
                        / output_name
                    )

                    mask_output = (
                        output_mask_dir
                        / output_name
                    )

                    Image.fromarray(
                        image_crop
                    ).save(
                        image_output
                    )

                    Image.fromarray(
                        mask_crop.astype(
                            np.uint8
                        )
                    ).save(
                        mask_output
                    )

                    total_crops += 1
                    region_crops += 1

                total_pairs += 1
                region_pairs += 1

            except Exception as e:

                failed += 1

                print(
                    f"\nERROR: "
                    f"{image_path.name}"
                )

                print(
                    f"       {e}"
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                index == 1
                or index % 100 == 0
                or index == len(image_files)
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                print(
                    f"\r{region}: "
                    f"{index}/{len(image_files)} "
                    f"| pairs={region_pairs} "
                    f"| crops={region_crops} "
                    f"| elapsed={elapsed:.1f}s",
                    end=""
                )

        print()

        print(
            f"{region} complete: "
            f"{region_pairs} pairs -> "
            f"{region_crops} crops"
        )

    elapsed = (
        time.time()
        - start_time
    )

    return {
        "source_split": source_split,
        "output_split": output_split,
        "source_images": total_source_images,
        "complete_pairs": total_pairs,
        "crops": total_crops,
        "failed": failed,
        "seconds": round(
            elapsed,
            2
        )
    }


# ============================================================
# PROCESS TRAIN
# ============================================================

train_stats = process_split(
    "train",
    "Train"
)


# ============================================================
# PROCESS VALIDATION
# ============================================================

val_stats = process_split(
    "val",
    "Val"
)


# ============================================================
# VERIFY OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING OUTPUT")
print("=" * 70)

verification = {}

for split in ["train", "val"]:

    image_dir = (
        PROCESSED_ROOT
        / split
        / "images"
    )

    mask_dir = (
        PROCESSED_ROOT
        / split
        / "masks"
    )

    images = sorted(
        image_dir.glob("*.png")
    )

    masks = sorted(
        mask_dir.glob("*.png")
    )

    image_names = {
        p.name
        for p in images
    }

    mask_names = {
        p.name
        for p in masks
    }

    common = (
        image_names
        &
        mask_names
    )

    missing_masks = (
        image_names
        -
        mask_names
    )

    missing_images = (
        mask_names
        -
        image_names
    )

    invalid_images = 0
    invalid_masks = 0
    class_values = set()

    # --------------------------------------------------------
    # Validate images
    # --------------------------------------------------------

    for image_path in images:

        try:

            image = np.array(
                Image.open(
                    image_path
                )
            )

            if image.shape != (
                512,
                512,
                3
            ):

                invalid_images += 1

        except Exception:

            invalid_images += 1

    # --------------------------------------------------------
    # Validate masks
    # --------------------------------------------------------

    for mask_path in masks:

        try:

            mask = np.array(
                Image.open(
                    mask_path
                )
            )

            if mask.shape != (
                512,
                512
            ):

                invalid_masks += 1

            class_values.update(
                np.unique(mask).tolist()
            )

        except Exception:

            invalid_masks += 1

    verification[split] = {

        "images": len(images),

        "masks": len(masks),

        "complete_pairs": len(common),

        "missing_masks": len(
            missing_masks
        ),

        "missing_images": len(
            missing_images
        ),

        "invalid_images": invalid_images,

        "invalid_masks": invalid_masks,

        "classes": sorted(
            int(x)
            for x in class_values
        )
    }

    print(
        f"\n{split.upper()}"
    )

    print(
        "Images:",
        len(images)
    )

    print(
        "Masks:",
        len(masks)
    )

    print(
        "Complete pairs:",
        len(common)
    )

    print(
        "Missing masks:",
        len(missing_masks)
    )

    print(
        "Missing images:",
        len(missing_images)
    )

    print(
        "Invalid images:",
        invalid_images
    )

    print(
        "Invalid masks:",
        invalid_masks
    )

    print(
        "Classes:",
        sorted(
            int(x)
            for x in class_values
        )
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "dataset": "LoveDA",

    "original_image_size": [
        1024,
        1024
    ],

    "crop_size": [
        CROP_SIZE,
        CROP_SIZE
    ],

    "crops_per_image": 4,

    "regions": REGIONS,

    "train": train_stats,

    "validation": val_stats,

    "verification": verification
}


metadata_path = (
    PROCESSED_ROOT
    / "metadata.json"
)

with open(
    metadata_path,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("LOVEDA PREPROCESSING COMPLETE")
print("=" * 70)

print(
    "\nTRAIN"
)

print(
    "Source images:",
    train_stats["source_images"]
)

print(
    "Complete pairs:",
    train_stats["complete_pairs"]
)

print(
    "512x512 crops:",
    train_stats["crops"]
)

print(
    "Failed:",
    train_stats["failed"]
)


print(
    "\nVALIDATION"
)

print(
    "Source images:",
    val_stats["source_images"]
)

print(
    "Complete pairs:",
    val_stats["complete_pairs"]
)

print(
    "512x512 crops:",
    val_stats["crops"]
)

print(
    "Failed:",
    val_stats["failed"]
)


print(
    "\nMetadata saved:"
)

print(
    metadata_path
)

print(
    "\n" + "=" * 70
)