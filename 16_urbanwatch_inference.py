# ============================================================
# URBANWATCH — UNIFIED INFERENCE
# ============================================================

from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw

import torch

from torchvision.models import resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation import DeepLabV3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

# This script is located at:
#
# UrbanWatch/
# └── results/
#     └── 16_urbanwatch_inference.py
#
# Therefore:
#
# parent        = UrbanWatch/results
# parent.parent = UrbanWatch

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# YOLO MODEL
# ------------------------------------------------------------

YOLO_MODEL = (
    ROOT
    / "results"
    / "yolo_training"
    / "spacenet_building_detector-4"
    / "weights"
    / "best.pt"
)


# ------------------------------------------------------------
# LOVEDA MODEL
# ------------------------------------------------------------

LOVEDA_MODEL = (
    ROOT
    / "results"
    / "loveda_segmentation_resnet18"
    / "best_model.pt"
)


# ------------------------------------------------------------
# LOVEDA INPUT IMAGES
# ------------------------------------------------------------

INPUT_DIR = (
    ROOT
    / "data"
    / "loveda"
    / "val"
    / "images"
)


# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "results"
    / "urbanwatch_inference"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# VISUALIZATION DIRECTORY
# ------------------------------------------------------------

VIS_DIR = (
    OUTPUT_DIR
    / "visualizations"
)


VIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# LoveDA
NUM_CLASSES = 8


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("URBANWATCH — UNIFIED INFERENCE")
print("=" * 70)

print()
print("Device:", DEVICE)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# PRINT PATHS
# ============================================================

print()
print("YOLO model:")
print(YOLO_MODEL)

print(
    "Exists:",
    YOLO_MODEL.exists()
)


print()
print("LoveDA model:")
print(LOVEDA_MODEL)

print(
    "Exists:",
    LOVEDA_MODEL.exists()
)


print()
print("Input directory:")
print(INPUT_DIR)

print(
    "Exists:",
    INPUT_DIR.exists()
)


print()
print("Output directory:")
print(OUTPUT_DIR)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not YOLO_MODEL.exists():

    raise FileNotFoundError(
        "\nYOLO model not found:\n"
        f"{YOLO_MODEL}"
    )


if not LOVEDA_MODEL.exists():

    raise FileNotFoundError(
        "\nLoveDA model not found:\n"
        f"{LOVEDA_MODEL}"
    )


if not INPUT_DIR.exists():

    raise FileNotFoundError(
        "\nLoveDA input directory not found:\n"
        f"{INPUT_DIR}"
    )


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("=" * 70)
print("LOADING YOLO")
print("=" * 70)


yolo_model = YOLO(
    str(YOLO_MODEL)
)


print()
print("YOLO loaded successfully.")


# ============================================================
# LOAD LOVEDA
# ============================================================

print()
print("=" * 70)
print("LOADING LOVEDA")
print("=" * 70)


# ------------------------------------------------------------
# IMPORTANT
#
# The trained LoveDA checkpoint uses:
#
#     DeepLabV3
#         |
#         └── ResNet18
#
# The checkpoint does NOT contain an auxiliary classifier.
#
# Therefore we must recreate exactly that architecture.
# ------------------------------------------------------------


# ------------------------------------------------------------
# Create ResNet18 backbone
# ------------------------------------------------------------

backbone = resnet18(
    weights=None
)


# ------------------------------------------------------------
# Extract only layer4
#
# layer4 produces 512 feature channels in ResNet18.
# ------------------------------------------------------------

backbone = IntermediateLayerGetter(
    backbone,
    return_layers={
        "layer4": "out"
    }
)


# ------------------------------------------------------------
# Create DeepLabV3
#
# No aux_classifier.
# ------------------------------------------------------------

loveda_model = DeepLabV3(
    backbone,
    classifier=DeepLabHead(
        512,
        NUM_CLASSES
    ),
    aux_classifier=None
)


# ============================================================
# LOAD LOVEDA CHECKPOINT
# ============================================================

checkpoint = torch.load(
    LOVEDA_MODEL,
    map_location=DEVICE,
    weights_only=False
)


# ------------------------------------------------------------
# Handle checkpoint formats
# ------------------------------------------------------------

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

loveda_model.load_state_dict(
    state_dict
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

loveda_model = loveda_model.to(
    DEVICE
)


loveda_model.eval()


print()
print("LoveDA loaded successfully.")


# ============================================================
# LOVEDA INFERENCE FUNCTION
# ============================================================

def predict_loveda(image):

    """
    Run semantic segmentation using
    DeepLabV3 + ResNet18.
    """

    # --------------------------------------------------------
    # Original image dimensions
    # --------------------------------------------------------

    original_size = image.size


    # --------------------------------------------------------
    # Convert image to RGB
    # --------------------------------------------------------

    image_array = np.array(
        image.convert("RGB")
    )


    # --------------------------------------------------------
    # Resize to model input
    # --------------------------------------------------------

    resized = Image.fromarray(
        image_array
    ).resize(
        (512, 512),
        Image.Resampling.BILINEAR
    )


    # --------------------------------------------------------
    # Convert to float32 [0, 1]
    # --------------------------------------------------------

    array = (
        np.array(resized)
        .astype(np.float32)
        / 255.0
    )


    # --------------------------------------------------------
    # HWC → CHW
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        array.transpose(
            2,
            0,
            1
        )
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )


    # --------------------------------------------------------
    # Move to GPU/CPU
    # --------------------------------------------------------

    tensor = tensor.to(
        DEVICE
    )


    # ========================================================
    # MODEL INFERENCE
    # ========================================================

    with torch.no_grad():

        if torch.cuda.is_available():

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                output = loveda_model(
                    tensor
                )["out"]

        else:

            output = loveda_model(
                tensor
            )["out"]


    # ========================================================
    # LOGITS → CLASS PREDICTION
    # ========================================================

    prediction = torch.argmax(
        output,
        dim=1
    )[0]


    # --------------------------------------------------------
    # GPU → CPU
    # --------------------------------------------------------

    prediction = (
        prediction
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


    # --------------------------------------------------------
    # Resize segmentation back to original size
    # --------------------------------------------------------

    prediction = np.array(
        Image.fromarray(
            prediction
        ).resize(
            original_size,
            Image.Resampling.NEAREST
        )
    )


    return prediction


# ============================================================
# BUILDING DETECTION
# ============================================================

def detect_buildings(image_path):

    """
    Run YOLO building detection.
    """

    results = yolo_model.predict(
        source=str(image_path),

        device=(
            0
            if torch.cuda.is_available()
            else "cpu"
        ),

        conf=0.35,

        verbose=False
    )


    result = results[0]


    boxes = []


    # ========================================================
    # EXTRACT DETECTIONS
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            # ------------------------------------------------
            # Bounding box coordinates
            # ------------------------------------------------

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = float(
                box.conf[0]
                .cpu()
                .item()
            )


            # ------------------------------------------------
            # Store detection
            # ------------------------------------------------

            boxes.append({

                "x1": xyxy[0],

                "y1": xyxy[1],

                "x2": xyxy[2],

                "y2": xyxy[3],

                "confidence": confidence

            })


    return boxes


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(
    image,
    segmentation,
    buildings,
    output_path
):

    """
    Create a visualization containing:

    - Original satellite image
    - LoveDA segmentation overlay
    - YOLO building bounding boxes
    """

    # --------------------------------------------------------
    # Ensure RGB
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )


    # --------------------------------------------------------
    # Create RGB segmentation image
    # --------------------------------------------------------

    segmentation_rgb = np.zeros(
        (
            segmentation.shape[0],
            segmentation.shape[1],
            3
        ),
        dtype=np.uint8
    )


    # ========================================================
    # LOVE DA CLASS PALETTE
    # ========================================================

    palette = np.array([

        [0, 0, 0],

        [255, 0, 0],

        [0, 255, 0],

        [0, 0, 255],

        [255, 255, 0],

        [255, 0, 255],

        [0, 255, 255],

        [255, 128, 0]

    ], dtype=np.uint8)


    # --------------------------------------------------------
    # Apply class colors
    # --------------------------------------------------------

    for cls in range(
        NUM_CLASSES
    ):

        segmentation_rgb[
            segmentation == cls
        ] = palette[cls]


    # --------------------------------------------------------
    # Convert segmentation to PIL
    # --------------------------------------------------------

    seg_image = Image.fromarray(
        segmentation_rgb
    )


    # --------------------------------------------------------
    # Resize segmentation to original image
    # --------------------------------------------------------

    seg_image = seg_image.resize(
        image.size,
        Image.Resampling.NEAREST
    )


    # ========================================================
    # BLEND SEGMENTATION
    # ========================================================

    combined = Image.blend(
        image,
        seg_image,
        alpha=0.35
    )


    # --------------------------------------------------------
    # Drawing object
    # --------------------------------------------------------

    draw = ImageDraw.Draw(
        combined
    )


    # ========================================================
    # DRAW YOLO BUILDING BOXES
    # ========================================================

    for building in buildings:

        draw.rectangle(
            [
                building["x1"],
                building["y1"],
                building["x2"],
                building["y2"]
            ],

            outline=(255, 255, 255),

            width=2
        )


    # ========================================================
    # SAVE
    # ========================================================

    combined.save(
        output_path
    )


# ============================================================
# RUN INFERENCE
# ============================================================

print()
print("=" * 70)
print("RUNNING URBANWATCH INFERENCE")
print("=" * 70)


# ------------------------------------------------------------
# Find PNG images
# ------------------------------------------------------------

images = sorted(
    INPUT_DIR.glob("*.png")
)


print()
print(
    "Images available:",
    len(images)
)


# ============================================================
# TEST WITH FIRST 10 IMAGES
# ============================================================

images = images[:10]


print()
print(
    "Images selected for inference:",
    len(images)
)


if len(images) == 0:

    raise RuntimeError(
        "\nNo PNG images found in:\n"
        f"{INPUT_DIR}"
    )


# ============================================================
# RESULTS STORAGE
# ============================================================

all_results = []


# ============================================================
# PROCESS IMAGES
# ============================================================

for index, image_path in enumerate(
    images,
    start=1
):

    print()
    print(
        f"[{index:02d}/{len(images):02d}] "
        f"{image_path.name}"
    )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    # ========================================================
    # LOVEDA SEGMENTATION
    # ========================================================

    print(
        "    Running LoveDA segmentation..."
    )


    segmentation = predict_loveda(
        image
    )


    # ========================================================
    # YOLO BUILDING DETECTION
    # ========================================================

    print(
        "    Running YOLO detection..."
    )


    buildings = detect_buildings(
        image_path
    )


    print(
        "    Buildings detected:",
        len(buildings)
    )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    visualization_path = (
        VIS_DIR
        / image_path.name
    )


    create_visualization(
        image,
        segmentation,
        buildings,
        visualization_path
    )


    print(
        "    Visualization saved:",
        visualization_path.name
    )


    # ========================================================
    # SEGMENTATION STATISTICS
    # ========================================================

    class_counts = {}


    for cls in range(
        NUM_CLASSES
    ):

        count = int(
            np.sum(
                segmentation == cls
            )
        )


        class_counts[
            str(cls)
        ] = count


    # ========================================================
    # STORE RESULT
    # ========================================================

    result = {

        "image":
            image_path.name,

        "building_count":
            len(buildings),

        "segmentation_classes":
            class_counts,

        "visualization":
            str(
                visualization_path
            )

    }


    all_results.append(
        result
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    OUTPUT_DIR
    / "urbanwatch_results.json"
)


with open(
    results_path,
    "w"
) as f:

    json.dump(
        all_results,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("URBANWATCH INFERENCE COMPLETE")
print("=" * 70)

print()
print(
    "Processed:",
    len(images)
)

print()
print(
    "Results:"
)

print(
    results_path
)

print()
print(
    "Visualizations:"
)

print(
    VIS_DIR
)

print()
print("=" * 70)