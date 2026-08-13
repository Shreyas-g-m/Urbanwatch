# ============================================================
# URBANWATCH — LoveDA SEMANTIC SEGMENTATION
# DeepLabV3 + ResNet18
#
# Optimized for:
# RTX 4070 Laptop GPU - 8 GB VRAM
# 16 GB System RAM
#
# IMPORTANT:
# - Existing LoveDA preprocessing is unchanged
# - 512x512 images
# - 8 classes
# - 10 epochs
# - RAM-safe validation
# - FP16 mixed precision
# ============================================================

import json
import time
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from torchvision.models._utils import IntermediateLayerGetter

from torchvision.models.segmentation.deeplabv3 import (
    DeepLabHead,
    DeepLabV3
)

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_ROOT = (
    ROOT
    / "data"
    / "loveda"
)

RESULTS_ROOT = (
    ROOT
    / "results"
    / "loveda_segmentation_resnet18"
)

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATA PATHS
# ============================================================

TRAIN_IMAGE_DIR = (
    DATA_ROOT
    / "train"
    / "images"
)

TRAIN_MASK_DIR = (
    DATA_ROOT
    / "train"
    / "masks"
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


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

NUM_CLASSES = 8

IMAGE_SIZE = 512

BATCH_SIZE = 4

# Keep 0 because your system has 16 GB RAM
NUM_WORKERS = 0

# User requested 10 epochs
EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

# Stop if mIoU does not improve for 3 epochs
PATIENCE = 3

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# CUDA PERFORMANCE
# ============================================================

if torch.cuda.is_available():

    torch.backends.cudnn.benchmark = True

    # RTX 40-series TF32 acceleration
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("URBANWATCH — LoveDA SEMANTIC SEGMENTATION")
print("DeepLabV3 + ResNet18")
print("=" * 70)

print()
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU VRAM: {gpu_memory:.1f} GB"
    )

print()
print("Classes:", NUM_CLASSES)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Workers:", NUM_WORKERS)
print("Epochs:", EPOCHS)
print("Early stopping patience:", PATIENCE)


# ============================================================
# DATASET
# ============================================================

class LoveDADataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir
    ):

        self.image_dir = Path(image_dir)

        self.mask_dir = Path(mask_dir)

        self.images = sorted(
            self.image_dir.glob("*.png")
        )

        self.pairs = []

        for image_path in self.images:

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

        print(
            f"Dataset: {self.image_dir}"
        )

        print(
            f"Complete pairs: {len(self.pairs)}"
        )


    def __len__(self):

        return len(self.pairs)


    def __getitem__(self, index):

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
        # MASK
        # ----------------------------------------------------

        mask = np.array(
            Image.open(
                mask_path
            )
        )

        mask = torch.from_numpy(
            mask.astype(
                np.int64
            )
        )


        return image, mask


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)


train_dataset = LoveDADataset(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR
)


val_dataset = LoveDADataset(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR
)


print()

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# RESNET18 BACKBONE
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL")
print("=" * 70)


# ------------------------------------------------------------
# Load pretrained ResNet18
# ------------------------------------------------------------

resnet = resnet18(
    weights=ResNet18_Weights.DEFAULT
)


# ------------------------------------------------------------
# ResNet18 feature extraction
#
# layer4 output = 512 channels
# layer1 output = 64 channels
# ------------------------------------------------------------

return_layers = {
    "layer4": "out",
    "layer1": "aux"
}


backbone = IntermediateLayerGetter(
    resnet,
    return_layers=return_layers
)


# ------------------------------------------------------------
# ResNet18 backbone channel counts
# ------------------------------------------------------------

backbone.out_channels = 512


# ------------------------------------------------------------
# DeepLabV3 head
# ------------------------------------------------------------

classifier = DeepLabHead(
    512,
    NUM_CLASSES
)


# ------------------------------------------------------------
# Create DeepLabV3
# ------------------------------------------------------------

model = DeepLabV3(
    backbone,
    classifier
)


model = model.to(
    DEVICE
)


# ============================================================
# MODEL INFORMATION
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(
    "Model: DeepLabV3-ResNet18"
)

print(
    f"Parameters: {total_parameters:,}"
)

print(
    f"Trainable parameters: {trainable_parameters:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# MIXED PRECISION
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ============================================================
# MEMORY-SAFE METRICS
# ============================================================

def update_confusion_matrix(
    confusion_matrix,
    predictions,
    targets
):

    predictions = predictions.reshape(-1)

    targets = targets.reshape(-1)


    encoded = (
        targets * NUM_CLASSES
        + predictions
    )


    counts = torch.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES
    )


    confusion_matrix += (
        counts
        .reshape(
            NUM_CLASSES,
            NUM_CLASSES
        )
        .cpu()
        .numpy()
    )


def calculate_metrics_from_confusion(
    confusion_matrix
):

    ious = []

    dices = []


    # --------------------------------------------------------
    # PIXEL ACCURACY
    # --------------------------------------------------------

    total_correct = np.trace(
        confusion_matrix
    )

    total_pixels = (
        confusion_matrix.sum()
    )

    if total_pixels > 0:

        accuracy = (
            total_correct
            / total_pixels
        )

    else:

        accuracy = 0.0


    # --------------------------------------------------------
    # PER-CLASS METRICS
    # --------------------------------------------------------

    for cls in range(NUM_CLASSES):

        true_positive = (
            confusion_matrix[cls, cls]
        )

        predicted_count = (
            confusion_matrix[:, cls].sum()
        )

        target_count = (
            confusion_matrix[cls, :].sum()
        )


        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        union = (
            predicted_count
            + target_count
            - true_positive
        )


        if union > 0:

            iou = (
                true_positive
                / union
            )

            ious.append(iou)


        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        denominator = (
            predicted_count
            + target_count
        )


        if denominator > 0:

            dice = (
                2.0
                * true_positive
                / denominator
            )

            dices.append(dice)


    mean_iou = (
        float(np.mean(ious))
        if ious
        else 0.0
    )


    mean_dice = (
        float(np.mean(dices))
        if dices
        else 0.0
    )


    return (
        mean_iou,
        mean_dice,
        float(accuracy)
    )


# ============================================================
# TRAINING STATE
# ============================================================

best_iou = 0.0

best_dice = 0.0

epochs_without_improvement = 0

history = []


# ============================================================
# START TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


training_start = time.time()


# ============================================================
# EPOCH LOOP
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start = time.time()


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0.0


    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch:03d}/{EPOCHS} [TRAIN]",
        leave=False,
        dynamic_ncols=True
    )


    for images, masks in train_progress:


        # ----------------------------------------------------
        # GPU TRANSFER
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
        # CLEAR GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # FORWARD
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

            loss = criterion(
                logits,
                masks
            )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )


        scaler.update()


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        train_loss += (
            loss.item()
            * images.size(0)
        )


        train_progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss /= len(
        train_dataset
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0


    # --------------------------------------------------------
    # ONLY 8x8 VALUES STORED
    # --------------------------------------------------------

    confusion_matrix = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=np.int64
    )


    with torch.inference_mode():


        val_progress = tqdm(
            val_loader,
            desc=f"Epoch {epoch:03d}/{EPOCHS} [VAL]",
            leave=False,
            dynamic_ncols=True
        )


        for images, masks in val_progress:


            # ------------------------------------------------
            # GPU TRANSFER
            # ------------------------------------------------

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )


            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
            ):

                outputs = model(
                    images
                )

                logits = outputs["out"]

                loss = criterion(
                    logits,
                    masks
                )


            val_loss += (
                loss.item()
                * images.size(0)
            )


            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            predictions = torch.argmax(
                logits,
                dim=1
            )


            # ------------------------------------------------
            # UPDATE CONFUSION MATRIX
            # ------------------------------------------------

            update_confusion_matrix(
                confusion_matrix,
                predictions,
                masks
            )


    # ========================================================
    # METRICS
    # ========================================================

    val_loss /= len(
        val_dataset
    )


    val_iou, val_dice, val_accuracy = (
        calculate_metrics_from_confusion(
            confusion_matrix
        )
    )


    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler.step(
        val_iou
    )


    # ========================================================
    # TIMING
    # ========================================================

    epoch_time = (
        time.time()
        - epoch_start
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # ========================================================
    # HISTORY
    # ========================================================

    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "val_loss": val_loss,

        "mIoU": val_iou,

        "mean_Dice": val_dice,

        "pixel_accuracy": val_accuracy,

        "learning_rate": current_lr,

        "seconds": epoch_time

    })


    # ========================================================
    # RESULTS
    # ========================================================

    print()

    print(
        f"Epoch {epoch:03d}/{EPOCHS}"
    )

    print(
        f"Train Loss:       {train_loss:.4f}"
    )

    print(
        f"Val Loss:         {val_loss:.4f}"
    )

    print(
        f"mIoU:             {val_iou:.4f}"
    )

    print(
        f"Mean Dice/F1:     {val_dice:.4f}"
    )

    print(
        f"Pixel Accuracy:   {val_accuracy:.4f}"
    )

    print(
        f"Learning Rate:    {current_lr:.6f}"
    )

    print(
        f"Epoch Time:       {epoch_time / 60:.2f} min"
    )


    # ========================================================
    # GPU MEMORY
    # ========================================================

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 3)
        )

        print(
            f"GPU Memory: "
            f"{allocated:.2f} GB allocated / "
            f"{reserved:.2f} GB reserved"
        )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if val_iou > best_iou:

        best_iou = val_iou

        best_dice = val_dice

        epochs_without_improvement = 0


        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_iou":
                best_iou,

            "best_dice":
                best_dice,

            "num_classes":
                NUM_CLASSES,

            "image_size":
                IMAGE_SIZE,

            "model_name":
                "DeepLabV3-ResNet18",

            "class_ids":
                list(range(NUM_CLASSES))

        }


        best_path = (
            RESULTS_ROOT
            / "best_model.pt"
        )


        torch.save(
            checkpoint,
            best_path
        )


        print()

        print(
            "NEW BEST MODEL"
        )

        print(
            f"Best mIoU: {best_iou:.4f}"
        )

        print(
            f"Best Dice: {best_dice:.4f}"
        )

        print(
            "Saved:",
            best_path
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()

        print(
            f"Early stopping after "
            f"{epoch} epochs."
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = (
    RESULTS_ROOT
    / "training_history.json"
)


with open(
    history_path,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

total_training_time = (
    time.time()
    - training_start
)


print()

print("=" * 70)
print("LOVEDA TRAINING COMPLETE")
print("=" * 70)

print()

print(
    f"Best mIoU: {best_iou:.4f}"
)

print(
    f"Best Mean Dice/F1: {best_dice:.4f}"
)

print()

print(
    f"Total training time: "
    f"{total_training_time / 3600:.2f} hours"
)

print()

print(
    "Best model:"
)

print(
    RESULTS_ROOT
    / "best_model.pt"
)

print()

print(
    "Training history:"
)

print(
    history_path
)

print()

print("=" * 70)