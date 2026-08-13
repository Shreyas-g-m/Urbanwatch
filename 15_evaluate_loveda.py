# ============================================================
# URBANWATCH — LoveDA FINAL EVALUATION
# DeepLabV3 + ResNet18
# ============================================================
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import (
    DeepLabHead,
    DeepLabV3
)

import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_ROOT = (
    ROOT
    / "data"
    / "loveda"
)

VAL_IMAGE_DIR = (
    DATA_ROOT
    / "val"
    / "images"
)

VAL_MASK_DIR = (
    DATA_ROOT
    / "val"
    / "masks"
)

# ------------------------------------------------------------
# ResNet18 trained checkpoint
# ------------------------------------------------------------

MODEL_PATH = (
    ROOT
    / "results"
    / "loveda_segmentation_resnet18"
    / "best_model.pt"
)

# ------------------------------------------------------------
# Evaluation output
# ------------------------------------------------------------

RESULTS_ROOT = (
    ROOT
    / "results"
    / "loveda_evaluation_resnet18"
)

PREDICTION_DIR = (
    RESULTS_ROOT
    / "predictions"
)

VISUALIZATION_DIR = (
    RESULTS_ROOT
    / "visualizations"
)


RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VISUALIZATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 8

BATCH_SIZE = 4

NUM_WORKERS = 0

IMAGE_SIZE = 512


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Background",
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
    4: "Class 4",
    5: "Class 5",
    6: "Class 6",
    7: "Class 7",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("URBANWATCH — LoveDA FINAL EVALUATION")
print("DeepLabV3 + ResNet18")
print("=" * 70)

print()

print("Model:")
print(MODEL_PATH)

print(
    "Exists:",
    MODEL_PATH.exists()
)

print()

print("Validation dataset:")
print(VAL_IMAGE_DIR)

print(
    "Exists:",
    VAL_IMAGE_DIR.exists()
)

print()

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


# ============================================================
# DATASET
# ============================================================

class LoveDADataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir
    ):

        self.image_dir = Path(
            image_dir
        )

        self.mask_dir = Path(
            mask_dir
        )

        self.pairs = []


        for image_path in sorted(
            self.image_dir.glob("*.png")
        ):

            mask_path = (
                self.mask_dir
                / image_path.name
            )

            if mask_path.exists():

                self.pairs.append(
                    (
                        image_path,
                        mask_path
                    )
                )


    def __len__(self):

        return len(
            self.pairs
        )


    def __getitem__(
        self,
        index
    ):

        image_path, mask_path = (
            self.pairs[index]
        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image = np.array(
            Image.open(
                image_path
            ).convert("RGB")
        )


        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        mask = np.array(
            Image.open(
                mask_path
            )
        )


        # ----------------------------------------------------
        # NORMALIZE IMAGE
        # ----------------------------------------------------

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )


        # HWC -> CHW

        image = torch.from_numpy(
            image.transpose(
                2,
                0,
                1
            )
        ).float()


        # ----------------------------------------------------
        # MASK -> LONG
        # ----------------------------------------------------

        mask = torch.from_numpy(
            mask.astype(
                np.int64
            )
        )


        return (
            image,
            mask,
            image_path.name
        )


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING VALIDATION DATA")
print("=" * 70)


dataset = LoveDADataset(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print()

print(
    "Validation pairs:",
    len(dataset)
)


# ============================================================
# CREATE MODEL — DEEPLABV3 + RESNET18
# ============================================================

print()
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)


# ------------------------------------------------------------
# ResNet18 backbone
#
# This MUST match the architecture used during training.
# ------------------------------------------------------------

resnet = resnet18(
    weights=None
)


# ------------------------------------------------------------
# Extract ResNet18 feature layers
# ------------------------------------------------------------

return_layers = {
    "layer4": "out",
    "layer1": "aux"
}


backbone = IntermediateLayerGetter(
    resnet,
    return_layers=return_layers
)


# ResNet18 layer4 has 512 output channels

backbone.out_channels = 512


# ------------------------------------------------------------
# DeepLabV3 segmentation classifier
# ------------------------------------------------------------

classifier = DeepLabHead(
    512,
    NUM_CLASSES
)


# ------------------------------------------------------------
# Create complete DeepLabV3 model
# ------------------------------------------------------------

model = DeepLabV3(
    backbone,
    classifier
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


# ------------------------------------------------------------
# Handle checkpoint format
# ------------------------------------------------------------

if (
    isinstance(
        checkpoint,
        dict
    )
    and "model_state_dict" in checkpoint
):

    state_dict = (
        checkpoint[
            "model_state_dict"
        ]
    )

else:

    state_dict = checkpoint


# ------------------------------------------------------------
# Load trained weights
# ------------------------------------------------------------

model.load_state_dict(
    state_dict,
    strict=True
)


model = model.to(
    DEVICE
)

model.eval()


print()

print(
    "Model: DeepLabV3-ResNet18"
)

print(
    "Model loaded successfully."
)


# ------------------------------------------------------------
# Print checkpoint information
# ------------------------------------------------------------

if isinstance(
    checkpoint,
    dict
):

    if "epoch" in checkpoint:

        print(
            "Best checkpoint epoch:",
            checkpoint["epoch"]
        )


    if "best_iou" in checkpoint:

        print(
            "Training best mIoU:",
            f"{checkpoint['best_iou']:.4f}"
        )


    if "best_dice" in checkpoint:

        print(
            "Training best Dice:",
            f"{checkpoint['best_dice']:.4f}"
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix = np.zeros(
    (
        NUM_CLASSES,
        NUM_CLASSES
    ),
    dtype=np.int64
)


# ============================================================
# RUN EVALUATION
# ============================================================

print()
print("=" * 70)
print("RUNNING FINAL VALIDATION")
print("=" * 70)


total_correct = 0

total_pixels = 0

sample_counter = 0


# ------------------------------------------------------------
# Memory-safe evaluation
# ------------------------------------------------------------

with torch.inference_mode():

    progress = tqdm(
        loader,
        desc="Evaluating",
        dynamic_ncols=True
    )


    for (
        images,
        masks,
        filenames
    ) in progress:


        # ----------------------------------------------------
        # MOVE TO GPU
        # ----------------------------------------------------

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):

            outputs = model(
                images
            )

            logits = outputs["out"]


        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predictions = torch.argmax(
            logits,
            dim=1
        )


        # ----------------------------------------------------
        # PIXEL ACCURACY
        # ----------------------------------------------------

        total_correct += (
            predictions == masks
        ).sum().item()


        total_pixels += (
            masks.numel()
        )


        # ----------------------------------------------------
        # MOVE ONLY THIS BATCH TO CPU
        # ----------------------------------------------------

        pred_np = (
            predictions
            .cpu()
            .numpy()
        )

        mask_np = (
            masks
            .cpu()
            .numpy()
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        for (
            pred_image,
            true_image
        ) in zip(
            pred_np,
            mask_np
        ):

            true_flat = (
                true_image
                .reshape(-1)
            )

            pred_flat = (
                pred_image
                .reshape(-1)
            )


            # ------------------------------------------------
            # VALID CLASS IDS
            # ------------------------------------------------

            valid = (
                (true_flat >= 0)
                &
                (true_flat < NUM_CLASSES)
                &
                (pred_flat >= 0)
                &
                (pred_flat < NUM_CLASSES)
            )


            indices = (
                NUM_CLASSES
                * true_flat[valid]
                + pred_flat[valid]
            )


            counts = np.bincount(
                indices,
                minlength=(
                    NUM_CLASSES
                    * NUM_CLASSES
                )
            )


            confusion_matrix += (
                counts.reshape(
                    NUM_CLASSES,
                    NUM_CLASSES
                )
            )


        # ====================================================
        # SAVE PREDICTIONS
        # ====================================================

        for i in range(
            len(filenames)
        ):

            prediction = (
                pred_np[i]
                .astype(
                    np.uint8
                )
            )


            prediction_path = (
                PREDICTION_DIR
                / filenames[i]
            )


            Image.fromarray(
                prediction
            ).save(
                prediction_path
            )


            # =================================================
            # SAVE VISUALIZATIONS
            # =================================================

            if sample_counter < 20:


                original_image = (
                    images[i]
                    .cpu()
                    .numpy()
                    .transpose(
                        1,
                        2,
                        0
                    )
                )


                original_image = np.clip(
                    original_image,
                    0,
                    1
                )


                ground_truth = (
                    mask_np[i]
                )


                fig, axes = plt.subplots(
                    1,
                    3,
                    figsize=(15, 5)
                )


                # ------------------------------------------------
                # Original
                # ------------------------------------------------

                axes[0].imshow(
                    original_image
                )

                axes[0].set_title(
                    "Image"
                )


                # ------------------------------------------------
                # Ground truth
                # ------------------------------------------------

                axes[1].imshow(
                    ground_truth
                )

                axes[1].set_title(
                    "Ground Truth"
                )


                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------

                axes[2].imshow(
                    prediction
                )

                axes[2].set_title(
                    "Prediction"
                )


                for ax in axes:

                    ax.axis(
                        "off"
                    )


                plt.tight_layout()


                visualization_path = (
                    VISUALIZATION_DIR
                    / f"sample_{sample_counter:02d}.png"
                )


                plt.savefig(
                    visualization_path,
                    dpi=150,
                    bbox_inches="tight"
                )


                plt.close()


                sample_counter += 1


# ============================================================
# CALCULATE METRICS
# ============================================================

print()
print("=" * 70)
print("CALCULATING METRICS")
print("=" * 70)


ious = []

dices = []

per_class = {}


# ============================================================
# PER-CLASS METRICS
# ============================================================

for cls in range(
    NUM_CLASSES
):


    # --------------------------------------------------------
    # TRUE POSITIVE
    # --------------------------------------------------------

    tp = confusion_matrix[
        cls,
        cls
    ]


    # --------------------------------------------------------
    # FALSE POSITIVE
    # --------------------------------------------------------

    fp = (
        confusion_matrix[
            :,
            cls
        ].sum()
        - tp
    )


    # --------------------------------------------------------
    # FALSE NEGATIVE
    # --------------------------------------------------------

    fn = (
        confusion_matrix[
            cls,
            :
        ].sum()
        - tp
    )


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    union = (
        tp
        + fp
        + fn
    )


    if union > 0:

        iou = (
            tp
            / union
        )

        ious.append(
            iou
        )

    else:

        iou = None


    # --------------------------------------------------------
    # Dice / F1
    # --------------------------------------------------------

    denominator = (
        2 * tp
        + fp
        + fn
    )


    if denominator > 0:

        dice = (
            2 * tp
            / denominator
        )

        dices.append(
            dice
        )

    else:

        dice = None


    # --------------------------------------------------------
    # Class pixel accuracy
    # --------------------------------------------------------

    class_pixels = (
        confusion_matrix[
            cls,
            :
        ].sum()
    )


    class_accuracy = (
        tp / class_pixels
        if class_pixels > 0
        else None
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    per_class[str(cls)] = {

        "name":
            CLASS_NAMES[cls],

        "IoU":
            None
            if iou is None
            else float(iou),

        "Dice":
            None
            if dice is None
            else float(dice),

        "pixel_accuracy":
            None
            if class_accuracy is None
            else float(
                class_accuracy
            ),

        "ground_truth_pixels":
            int(class_pixels)
    }


# ============================================================
# GLOBAL METRICS
# ============================================================

mean_iou = (
    float(
        np.mean(
            ious
        )
    )
    if ious
    else 0.0
)


mean_dice = (
    float(
        np.mean(
            dices
        )
    )
    if dices
    else 0.0
)


pixel_accuracy = (
    total_correct
    / total_pixels
    if total_pixels > 0
    else 0.0
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL LOVEDA VALIDATION RESULTS")
print("=" * 70)


print()

print(
    f"Validation pairs : {len(dataset)}"
)

print(
    f"mIoU             : {mean_iou:.4f}"
)

print(
    f"Mean Dice / F1   : {mean_dice:.4f}"
)

print(
    f"Pixel Accuracy   : {pixel_accuracy:.4f}"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print()

print(
    "PER-CLASS RESULTS"
)

print(
    "-" * 70
)


for cls in range(
    NUM_CLASSES
):


    result = (
        per_class[
            str(cls)
        ]
    )


    iou = result["IoU"]

    dice = result["Dice"]


    iou_text = (
        f"{iou:.4f}"
        if iou is not None
        else "N/A"
    )


    dice_text = (
        f"{dice:.4f}"
        if dice is not None
        else "N/A"
    )


    print(
        f"{cls:>2} "
        f"{CLASS_NAMES[cls]:<12} "
        f"IoU: {iou_text} "
        f"Dice: {dice_text}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()

print(
    "CONFUSION MATRIX"
)

print(
    "-" * 70
)

print(
    confusion_matrix
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "dataset":
        "LoveDA",

    "model":
        "DeepLabV3-ResNet18",

    "checkpoint":
        str(MODEL_PATH),

    "validation_pairs":
        len(dataset),

    "image_size":
        IMAGE_SIZE,

    "num_classes":
        NUM_CLASSES,

    "mIoU":
        mean_iou,

    "mean_Dice_F1":
        mean_dice,

    "pixel_accuracy":
        pixel_accuracy,

    "per_class":
        per_class,

    "confusion_matrix":
        confusion_matrix.tolist(),

    "predictions_directory":
        str(PREDICTION_DIR),

    "visualizations_directory":
        str(VISUALIZATION_DIR)
}


# ============================================================
# SAVE JSON
# ============================================================

results_path = (
    RESULTS_ROOT
    / "evaluation_results.json"
)


with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print()

print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print()

print(
    "Results:"
)

print(
    results_path
)

print()

print(
    "Predictions:"
)

print(
    PREDICTION_DIR
)

print()

print(
    "Visualizations:"
)

print(
    VISUALIZATION_DIR
)

print()

print(
    "Final mIoU:",
    f"{mean_iou:.4f}"
)

print(
    "Final Dice/F1:",
    f"{mean_dice:.4f}"
)

print(
    "Final Pixel Accuracy:",
    f"{pixel_accuracy:.4f}"
)

print()

print("=" * 70)