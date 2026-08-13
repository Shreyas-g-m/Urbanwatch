# ============================================================
# URBANWATCH — LEVIR-CD+ CHANGE DETECTION
# Siamese U-Net
# ============================================================

import os

# Windows OpenMP compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import random
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_ROOT = ROOT / "data" / "levir_cd" / "split"

RESULTS_ROOT = ROOT / "results" / "change_detection"

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

CROP_SIZE = 512

BATCH_SIZE = 4

EPOCHS = 50

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

VAL_EVERY = 1

PATIENCE = 10


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# GPU SETTINGS
# ============================================================

if torch.cuda.is_available():

    torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print("GPU")
    print("=" * 70)

    print(torch.cuda.get_device_name(0))

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )


print("\nDevice:", DEVICE)


# ============================================================
# DATASET
# ============================================================

class LEVIRDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        crop_size=512,
        training=False
    ):

        self.root = Path(root)

        self.split = split

        self.crop_size = crop_size

        self.training = training

        self.a_dir = (
            self.root
            / split
            / "A"
        )

        self.b_dir = (
            self.root
            / split
            / "B"
        )

        self.label_dir = (
            self.root
            / split
            / "labels"
        )

        self.files = sorted(
            [
                p.name
                for p in self.a_dir.glob("*.png")
                if (
                    (self.b_dir / p.name).exists()
                    and
                    (self.label_dir / p.name).exists()
                )
            ]
        )

        if len(self.files) == 0:

            raise RuntimeError(
                f"No valid LEVIR pairs found in {self.a_dir}"
            )


    def __len__(self):

        return len(self.files)


    def _random_crop(
        self,
        a,
        b,
        mask
    ):

        h, w = mask.shape

        if h < self.crop_size or w < self.crop_size:

            raise ValueError(
                f"Image smaller than crop size: "
                f"{h}x{w}"
            )

        top = random.randint(
            0,
            h - self.crop_size
        )

        left = random.randint(
            0,
            w - self.crop_size
        )

        a = a[
            top:top + self.crop_size,
            left:left + self.crop_size
        ]

        b = b[
            top:top + self.crop_size,
            left:left + self.crop_size
        ]

        mask = mask[
            top:top + self.crop_size,
            left:left + self.crop_size
        ]

        return a, b, mask


    def _augment(
        self,
        a,
        b,
        mask
    ):

        # Horizontal flip
        if random.random() < 0.5:

            a = np.fliplr(a).copy()
            b = np.fliplr(b).copy()
            mask = np.fliplr(mask).copy()


        # Vertical flip
        if random.random() < 0.5:

            a = np.flipud(a).copy()
            b = np.flipud(b).copy()
            mask = np.flipud(mask).copy()


        # 90-degree rotation
        if random.random() < 0.5:

            k = random.randint(
                1,
                3
            )

            a = np.rot90(
                a,
                k
            ).copy()

            b = np.rot90(
                b,
                k
            ).copy()

            mask = np.rot90(
                mask,
                k
            ).copy()


        return a, b, mask


    def __getitem__(self, index):

        filename = self.files[index]

        a_path = self.a_dir / filename
        b_path = self.b_dir / filename
        label_path = self.label_dir / filename

        a = np.array(
            Image.open(a_path).convert("RGB"),
            dtype=np.float32
        )

        b = np.array(
            Image.open(b_path).convert("RGB"),
            dtype=np.float32
        )

        mask = np.array(
            Image.open(label_path).convert("L"),
            dtype=np.uint8
        )


        # Normalize images
        a /= 255.0
        b /= 255.0

        # Binary mask
        mask = (
            mask > 0
        ).astype(
            np.float32
        )


        # Training crop
        if self.training:

            a, b, mask = self._random_crop(
                a,
                b,
                mask
            )

            a, b, mask = self._augment(
                a,
                b,
                mask
            )

        else:

            # Validation crop
            # deterministic top-left crop
            a = a[
                :self.crop_size,
                :self.crop_size
            ]

            b = b[
                :self.crop_size,
                :self.crop_size
            ]

            mask = mask[
                :self.crop_size,
                :self.crop_size
            ]


        # HWC → CHW
        a = torch.from_numpy(
            a.transpose(2, 0, 1)
        ).float()

        b = torch.from_numpy(
            b.transpose(2, 0, 1)
        ).float()

        mask = torch.from_numpy(
            mask
        ).float().unsqueeze(0)


        return a, b, mask, filename


# ============================================================
# MODEL BLOCKS
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )


    def forward(self, x):

        return self.block(x)


class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.c1 = ConvBlock(
            3,
            32
        )

        self.c2 = ConvBlock(
            32,
            64
        )

        self.c3 = ConvBlock(
            64,
            128
        )

        self.c4 = ConvBlock(
            128,
            256
        )

        self.pool = nn.MaxPool2d(
            2
        )


    def forward(self, x):

        x1 = self.c1(x)

        x2 = self.c2(
            self.pool(x1)
        )

        x3 = self.c3(
            self.pool(x2)
        )

        x4 = self.c4(
            self.pool(x3)
        )

        return x1, x2, x3, x4


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels
        )


    def forward(
        self,
        x,
        skip
    ):

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [x, skip],
            dim=1
        )

        return self.conv(x)


# ============================================================
# SIAMESE U-NET
# ============================================================

class SiameseUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Shared encoder
        self.encoder = Encoder()

        # Decoder
        self.d3 = DecoderBlock(
            256,
            128,
            128
        )

        self.d2 = DecoderBlock(
            128,
            64,
            64
        )

        self.d1 = DecoderBlock(
            64,
            32,
            32
        )

        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )


    def forward(
        self,
        a,
        b
    ):

        a1, a2, a3, a4 = self.encoder(a)

        b1, b2, b3, b4 = self.encoder(b)

        # Feature differences
        x4 = torch.abs(
            a4 - b4
        )

        x3 = torch.abs(
            a3 - b3
        )

        x2 = torch.abs(
            a2 - b2
        )

        x1 = torch.abs(
            a1 - b1
        )

        x = self.d3(
            x4,
            x3
        )

        x = self.d2(
            x,
            x2
        )

        x = self.d1(
            x,
            x1
        )

        x = F.interpolate(
            x,
            size=a.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return self.final(x)


# ============================================================
# LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth


    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.flatten(
            1
        )

        targets = targets.flatten(
            1
        )

        intersection = (
            probabilities * targets
        ).sum(
            dim=1
        )

        dice = (
            2 * intersection
            + self.smooth
        ) / (
            probabilities.sum(dim=1)
            +
            targets.sum(dim=1)
            +
            self.smooth
        )

        return 1 - dice.mean()


class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()


    def forward(
        self,
        logits,
        targets
    ):

        return (
            0.5 * self.bce(
                logits,
                targets
            )
            +
            0.5 * self.dice(
                logits,
                targets
            )
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    targets,
    threshold=0.5
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    targets = targets.float()


    tp = (
        predictions * targets
    ).sum().item()

    fp = (
        predictions * (1 - targets)
    ).sum().item()

    fn = (
        (1 - predictions) * targets
    ).sum().item()

    tn = (
        (1 - predictions)
        *
        (1 - targets)
    ).sum().item()


    epsilon = 1e-7


    precision = (
        tp /
        (tp + fp + epsilon)
    )

    recall = (
        tp /
        (tp + fn + epsilon)
    )

    dice = (
        2 * tp /
        (
            2 * tp
            + fp
            + fn
            + epsilon
        )
    )

    iou = (
        tp /
        (
            tp
            + fp
            + fn
            + epsilon
        )
    )

    accuracy = (
        (tp + tn) /
        (
            tp
            + tn
            + fp
            + fn
            + epsilon
        )
    )


    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "accuracy": accuracy
    }


# ============================================================
# DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_dataset = LEVIRDataset(
    DATA_ROOT,
    "train",
    crop_size=CROP_SIZE,
    training=True
)

val_dataset = LEVIRDataset(
    DATA_ROOT,
    "val",
    crop_size=CROP_SIZE,
    training=False
)

print(
    "Training pairs:",
    len(train_dataset)
)

print(
    "Validation pairs:",
    len(val_dataset)
)


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
# MODEL
# ============================================================

print("\n" + "=" * 70)
print("CREATING MODEL")
print("=" * 70)

model = SiameseUNet().to(
    DEVICE
)

parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Parameters:",
    f"{parameters:,}"
)


# ============================================================
# OPTIMIZER
# ============================================================

criterion = CombinedLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# ============================================================
# AMP
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ============================================================
# TRAINING
# ============================================================

best_iou = 0.0

best_dice = 0.0

epochs_without_improvement = 0

history = []


print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(
    1,
    EPOCHS + 1
):


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    train_batches = 0


    for batch in train_loader:

        a, b, masks, _ = batch

        a = a.to(
            DEVICE,
            non_blocking=True
        )

        b = b.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):

            outputs = model(
                a,
                b
            )

            loss = criterion(
                outputs,
                masks
            )


        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()


        train_loss += loss.item()

        train_batches += 1


    train_loss /= max(
        train_batches,
        1
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    metric_totals = {
        "precision": 0.0,
        "recall": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "accuracy": 0.0
    }

    val_batches = 0


    with torch.no_grad():

        for batch in val_loader:

            a, b, masks, _ = batch

            a = a.to(
                DEVICE,
                non_blocking=True
            )

            b = b.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )


            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
            ):

                outputs = model(
                    a,
                    b
                )

                loss = criterion(
                    outputs,
                    masks
                )


            val_loss += loss.item()


            metrics = calculate_metrics(
                outputs,
                masks
            )


            for key in metric_totals:

                metric_totals[key] += metrics[key]


            val_batches += 1


    val_loss /= max(
        val_batches,
        1
    )


    for key in metric_totals:

        metric_totals[key] /= max(
            val_batches,
            1
        )


    scheduler.step(
        metric_totals["iou"]
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    epoch_result = {

        "epoch": epoch,

        "train_loss": train_loss,

        "val_loss": val_loss,

        **metric_totals,

        "learning_rate": optimizer.param_groups[0]["lr"]
    }


    history.append(
        epoch_result
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch:03d}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f}"
    )

    print(
        f"Precision:  {metric_totals['precision']:.4f}"
    )

    print(
        f"Recall:     {metric_totals['recall']:.4f}"
    )

    print(
        f"Dice/F1:    {metric_totals['dice']:.4f}"
    )

    print(
        f"IoU:        {metric_totals['iou']:.4f}"
    )

    print(
        f"Accuracy:   {metric_totals['accuracy']:.4f}"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    current_iou = metric_totals["iou"]


    if current_iou > best_iou:

        best_iou = current_iou

        best_dice = metric_totals["dice"]

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

            "config": {

                "crop_size":
                    CROP_SIZE,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "seed":
                    SEED
            }
        }


        torch.save(
            checkpoint,
            RESULTS_ROOT / "best_model.pt"
        )


        print(
            "✓ Best model saved."
        )


    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping after "
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

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Best IoU:",
    f"{best_iou:.4f}"
)

print(
    "Best Dice/F1:",
    f"{best_dice:.4f}"
)

print(
    "\nBest model:"
)

print(
    RESULTS_ROOT / "best_model.pt"
)

print(
    "\nTraining history:"
)

print(
    history_path
)