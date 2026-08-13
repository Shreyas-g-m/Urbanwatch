# ============================================================
# URBANWATCH — LEVIR-CD+ FINAL TEST EVALUATION
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS / CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

TEST_ROOT = ROOT / "data" / "levir_cd" / "test"

MODEL_PATH = (
    ROOT
    / "results"
    / "change_detection"
    / "best_model.pt"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "change_detection"
    / "test_evaluation"
)

PREDICTION_DIR = OUTPUT_ROOT / "predictions"
VIS_DIR = OUTPUT_ROOT / "visualizations"

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CROP_SIZE = 512
BATCH_SIZE = 4
NUM_WORKERS = 0
THRESHOLD = 0.5


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("URBANWATCH — LEVIR-CD+ FINAL TEST EVALUATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)
print("Exists:", MODEL_PATH.exists())

print("\nTest dataset:")
print(TEST_ROOT)
print("Exists:", TEST_ROOT.exists())

print("\nDevice:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# ============================================================
# DATASET
# ============================================================

class LEVIRTestDataset(Dataset):

    def __init__(
        self,
        root,
        crop_size=512
    ):

        self.root = Path(root)

        self.crop_size = crop_size

        self.a_dir = self.root / "A"
        self.b_dir = self.root / "B"
        self.label_dir = self.root / "labels"

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
                "No complete test pairs found."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        a = np.array(
            Image.open(
                self.a_dir / filename
            ).convert("RGB"),
            dtype=np.float32
        )

        b = np.array(
            Image.open(
                self.b_dir / filename
            ).convert("RGB"),
            dtype=np.float32
        )

        mask = np.array(
            Image.open(
                self.label_dir / filename
            ).convert("L"),
            dtype=np.uint8
        )

        # Normalize
        a /= 255.0
        b /= 255.0

        # Binary mask
        mask = (
            mask > 0
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Deterministic center crop
        # ----------------------------------------------------

        h, w = mask.shape

        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2

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

        # HWC -> CHW
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
# MODEL
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
                3,
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
                3,
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

        self.c1 = ConvBlock(3, 32)
        self.c2 = ConvBlock(32, 64)
        self.c3 = ConvBlock(64, 128)
        self.c4 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

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


class SiameseUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder()

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
            1
        )

    def forward(
        self,
        a,
        b
    ):

        a1, a2, a3, a4 = self.encoder(a)

        b1, b2, b3, b4 = self.encoder(b)

        x4 = torch.abs(a4 - b4)
        x3 = torch.abs(a3 - b3)
        x2 = torch.abs(a2 - b2)
        x1 = torch.abs(a1 - b1)

        x = self.d3(x4, x3)
        x = self.d2(x, x2)
        x = self.d1(x, x1)

        x = F.interpolate(
            x,
            size=a.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return self.final(x)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

dataset = LEVIRTestDataset(
    TEST_ROOT,
    crop_size=CROP_SIZE
)

print(
    "Complete test pairs:",
    len(dataset)
)

if len(dataset) != 348:
    print(
        "WARNING: Expected 348 test pairs."
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

model = SiameseUNet().to(
    DEVICE
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Model loaded successfully."
)

if "epoch" in checkpoint:
    print(
        "Best checkpoint epoch:",
        checkpoint["epoch"]
    )

if "best_iou" in checkpoint:
    print(
        "Validation IoU:",
        f"{checkpoint['best_iou']:.4f}"
    )

if "best_dice" in checkpoint:
    print(
        "Validation Dice:",
        f"{checkpoint['best_dice']:.4f}"
    )


# ============================================================
# METRIC ACCUMULATORS
# ============================================================

TP = 0
TN = 0
FP = 0
FN = 0

visualizations_saved = 0
MAX_VISUALIZATIONS = 10


# ============================================================
# INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("RUNNING FINAL TEST")
print("=" * 70)

with torch.no_grad():

    for batch_idx, batch in enumerate(loader):

        a, b, masks, filenames = batch

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

            logits = model(
                a,
                b
            )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        TP += (
            predictions * masks
        ).sum().item()

        FP += (
            predictions * (1 - masks)
        ).sum().item()

        FN += (
            (1 - predictions) * masks
        ).sum().item()

        TN += (
            (1 - predictions)
            *
            (1 - masks)
        ).sum().item()


        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        predictions_cpu = (
            predictions
            .cpu()
            .numpy()
        )

        a_cpu = (
            a.cpu()
            .numpy()
        )

        b_cpu = (
            b.cpu()
            .numpy()
        )

        masks_cpu = (
            masks.cpu()
            .numpy()
        )


        for i, filename in enumerate(
            filenames
        ):

            pred = (
                predictions_cpu[i, 0]
                * 255
            ).astype(
                np.uint8
            )

            Image.fromarray(
                pred
            ).save(
                PREDICTION_DIR / filename
            )


            # ------------------------------------------------
            # Visualizations
            # ------------------------------------------------

            if (
                visualizations_saved
                <
                MAX_VISUALIZATIONS
            ):

                image_a = (
                    a_cpu[i]
                    .transpose(1, 2, 0)
                    * 255
                ).clip(
                    0,
                    255
                ).astype(
                    np.uint8
                )

                image_b = (
                    b_cpu[i]
                    .transpose(1, 2, 0)
                    * 255
                ).clip(
                    0,
                    255
                ).astype(
                    np.uint8
                )

                gt = (
                    masks_cpu[i, 0]
                    * 255
                ).astype(
                    np.uint8
                )

                prediction = pred

                # Create side-by-side image
                canvas = Image.new(
                    "RGB",
                    (
                        CROP_SIZE * 4,
                        CROP_SIZE
                    )
                )

                canvas.paste(
                    Image.fromarray(image_a),
                    (0, 0)
                )

                canvas.paste(
                    Image.fromarray(image_b),
                    (CROP_SIZE, 0)
                )

                canvas.paste(
                    Image.fromarray(gt).convert("RGB"),
                    (CROP_SIZE * 2, 0)
                )

                canvas.paste(
                    Image.fromarray(prediction).convert("RGB"),
                    (CROP_SIZE * 3, 0)
                )

                draw = ImageDraw.Draw(
                    canvas
                )

                draw.rectangle(
                    (0, 0, CROP_SIZE, 35),
                    fill="black"
                )

                draw.rectangle(
                    (
                        CROP_SIZE,
                        0,
                        CROP_SIZE * 2,
                        35
                    ),
                    fill="black"
                )

                draw.rectangle(
                    (
                        CROP_SIZE * 2,
                        0,
                        CROP_SIZE * 3,
                        35
                    ),
                    fill="black"
                )

                draw.rectangle(
                    (
                        CROP_SIZE * 3,
                        0,
                        CROP_SIZE * 4,
                        35
                    ),
                    fill="black"
                )

                draw.text(
                    (10, 10),
                    "TIME A",
                    fill="white"
                )

                draw.text(
                    (CROP_SIZE + 10, 10),
                    "TIME B",
                    fill="white"
                )

                draw.text(
                    (
                        CROP_SIZE * 2 + 10,
                        10
                    ),
                    "GROUND TRUTH",
                    fill="white"
                )

                draw.text(
                    (
                        CROP_SIZE * 3 + 10,
                        10
                    ),
                    "PREDICTION",
                    fill="white"
                )

                canvas.save(
                    VIS_DIR
                    / f"comparison_{visualizations_saved + 1}.png"
                )

                visualizations_saved += 1


        if (
            batch_idx == 0
            or
            (batch_idx + 1) % 10 == 0
            or
            (batch_idx + 1) == len(loader)
        ):

            print(
                f"[{batch_idx + 1:03d}/{len(loader):03d}] "
                f"processed"
            )


# ============================================================
# FINAL METRICS
# ============================================================

EPS = 1e-7

precision = (
    TP /
    (TP + FP + EPS)
)

recall = (
    TP /
    (TP + FN + EPS)
)

dice = (
    2 * TP /
    (
        2 * TP
        + FP
        + FN
        + EPS
    )
)

iou = (
    TP /
    (
        TP
        + FP
        + FN
        + EPS
    )
)

accuracy = (
    (TP + TN) /
    (
        TP
        + TN
        + FP
        + FN
        + EPS
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "dataset": "LEVIR-CD+",

    "test_pairs": len(dataset),

    "crop_size": CROP_SIZE,

    "threshold": THRESHOLD,

    "precision": precision,

    "recall": recall,

    "dice_f1": dice,

    "iou": iou,

    "pixel_accuracy": accuracy,

    "true_positive": TP,

    "true_negative": TN,

    "false_positive": FP,

    "false_negative": FN
}


summary_path = (
    OUTPUT_ROOT
    / "test_results.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL LEVIR-CD+ TEST RESULTS")
print("=" * 70)

print(
    f"Test pairs       : {len(dataset)}"
)

print(
    f"Precision        : {precision:.4f}"
)

print(
    f"Recall           : {recall:.4f}"
)

print(
    f"Dice / F1        : {dice:.4f}"
)

print(
    f"IoU              : {iou:.4f}"
)

print(
    f"Pixel Accuracy   : {accuracy:.4f}"
)

print("\nPredictions:")
print(PREDICTION_DIR)

print("\nVisualizations:")
print(VIS_DIR)

print("\nResults JSON:")
print(summary_path)# ============================================================
# URBANWATCH — LEVIR-CD+ CHANGE DETECTION INFERENCE
# ============================================================

from pathlib import Path
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

# This script is directly inside:
#
# C:\Users\shreyas\Documents\UrbanWatch\
#     17_change_detection_inference.py
#
# Therefore:
# parent = UrbanWatch

ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_PATH = (
    ROOT
    / "results"
    / "change_detection"
    / "best_model.pt"
)


# ------------------------------------------------------------
# LEVIR-CD+ TEST DATASET
# ------------------------------------------------------------

TEST_ROOT = (
    ROOT
    / "data"
    / "levir_cd"
    / "test"
)


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "change_detection_inference"
)


VIS_DIR = (
    OUTPUT_ROOT
    / "visualizations"
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


VIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
print("URBANWATCH — LEVIR-CD+ CHANGE DETECTION INFERENCE")
print("=" * 70)


print()
print("Model:")
print(MODEL_PATH)

print(
    "Exists:",
    MODEL_PATH.exists()
)


print()
print("Test dataset:")
print(TEST_ROOT)

print(
    "Exists:",
    TEST_ROOT.exists()
)


print()
print("Device:", DEVICE)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECK PATHS
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "\nChange detection model not found:\n"
        f"{MODEL_PATH}"
    )


if not TEST_ROOT.exists():

    raise FileNotFoundError(
        "\nLEVIR-CD+ test dataset not found:\n"
        f"{TEST_ROOT}"
    )


# ============================================================
# CONVOLUTION BLOCK
# ============================================================

class ConvBlock(nn.Module):

    """
    Two Conv-BN-ReLU layers.

    The checkpoint confirms:

        Conv weights exist
        Conv biases do NOT exist

    Therefore:

        bias=False

    is required for both convolutions.
    """

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()


        self.block = nn.Sequential(

            # ------------------------------------------------
            # First convolution
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Second convolution
            # ------------------------------------------------

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


    def forward(
        self,
        x
    ):

        return self.block(
            x
        )


# ============================================================
# DECODER BLOCK
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()


        self.conv = ConvBlock(
            in_channels,
            out_channels
        )


    def forward(
        self,
        x
    ):

        return self.conv(
            x
        )


# ============================================================
# CHANGE DETECTION MODEL
# ============================================================

class SiameseChangeDetector(nn.Module):

    """
    U-Net-style LEVIR-CD+ change detector.

    Architecture reconstructed from the checkpoint:

        Input:
            3 channels

        Encoder:
            c1: 3   -> 32
            c2: 32  -> 64
            c3: 64  -> 128
            c4: 128 -> 256

        Decoder:
            d3: 384 -> 128
            d2: 192 -> 64
            d1: 96  -> 32

        Final:
            32 -> 1

    The checkpoint also confirms that the convolution
    layers inside ConvBlock use bias=False.
    """

    def __init__(self):

        super().__init__()


        # ====================================================
        # ENCODER
        # ====================================================

        self.encoder = nn.ModuleDict({

            "c1": ConvBlock(
                3,
                32
            ),

            "c2": ConvBlock(
                32,
                64
            ),

            "c3": ConvBlock(
                64,
                128
            ),

            "c4": ConvBlock(
                128,
                256
            )
        })


        # ====================================================
        # DECODER
        # ====================================================

        self.d3 = DecoderBlock(
            384,
            128
        )


        self.d2 = DecoderBlock(
            192,
            64
        )


        self.d1 = DecoderBlock(
            96,
            32
        )


        # ====================================================
        # FINAL LAYER
        # ====================================================

        # IMPORTANT:
        # The checkpoint contains:
        #
        # final.weight
        # final.bias
        #
        # Therefore this layer keeps bias=True.

        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1,
            bias=True
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        image_a,
        image_b
    ):

        # ----------------------------------------------------
        # Temporal difference
        # ----------------------------------------------------

        x = torch.abs(
            image_a - image_b
        )


        # ====================================================
        # ENCODER
        # ====================================================

        c1 = self.encoder["c1"](
            x
        )


        c2_input = F.max_pool2d(
            c1,
            kernel_size=2
        )


        c2 = self.encoder["c2"](
            c2_input
        )


        c3_input = F.max_pool2d(
            c2,
            kernel_size=2
        )


        c3 = self.encoder["c3"](
            c3_input
        )


        c4_input = F.max_pool2d(
            c3,
            kernel_size=2
        )


        c4 = self.encoder["c4"](
            c4_input
        )


        # ====================================================
        # DECODER 3
        # ====================================================

        d3_up = F.interpolate(
            c4,
            size=c3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )


        d3_input = torch.cat(
            [
                d3_up,
                c3
            ],
            dim=1
        )


        d3 = self.d3(
            d3_input
        )


        # ====================================================
        # DECODER 2
        # ====================================================

        d2_up = F.interpolate(
            d3,
            size=c2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )


        d2_input = torch.cat(
            [
                d2_up,
                c2
            ],
            dim=1
        )


        d2 = self.d2(
            d2_input
        )


        # ====================================================
        # DECODER 1
        # ====================================================

        d1_up = F.interpolate(
            d2,
            size=c1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )


        d1_input = torch.cat(
            [
                d1_up,
                c1
            ],
            dim=1
        )


        d1 = self.d1(
            d1_input
        )


        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        output = self.final(
            d1
        )


        return output


# ============================================================
# DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(self):

        self.a_dir = (
            TEST_ROOT
            / "A"
        )


        self.b_dir = (
            TEST_ROOT
            / "B"
        )


        # IMPORTANT:
        #
        # Your actual folder is:
        #
        # test/labels
        #
        # NOT:
        #
        # test/label

        self.label_dir = (
            TEST_ROOT
            / "labels"
        )


        self.pairs = []


        # ====================================================
        # CHECK FOLDERS
        # ====================================================

        print()
        print("Dataset folders:")


        print(
            "A:",
            self.a_dir,
            "| Exists:",
            self.a_dir.exists()
        )


        print(
            "B:",
            self.b_dir,
            "| Exists:",
            self.b_dir.exists()
        )


        print(
            "Labels:",
            self.label_dir,
            "| Exists:",
            self.label_dir.exists()
        )


        if not self.a_dir.exists():

            raise FileNotFoundError(
                f"\nA folder not found:\n"
                f"{self.a_dir}"
            )


        if not self.b_dir.exists():

            raise FileNotFoundError(
                f"\nB folder not found:\n"
                f"{self.b_dir}"
            )


        if not self.label_dir.exists():

            raise FileNotFoundError(
                f"\nLabels folder not found:\n"
                f"{self.label_dir}"
            )


        # ====================================================
        # FIND COMPLETE PAIRS
        # ====================================================

        for path in sorted(
            self.a_dir.glob("*.png")
        ):

            filename = path.name


            b_path = (
                self.b_dir
                / filename
            )


            label_path = (
                self.label_dir
                / filename
            )


            if (
                b_path.exists()
                and label_path.exists()
            ):

                self.pairs.append(
                    (
                        path,
                        b_path,
                        label_path
                    )
                )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.pairs
        )


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        (
            a_path,
            b_path,
            label_path
        ) = self.pairs[index]


        # ----------------------------------------------------
        # IMAGE A
        # ----------------------------------------------------

        a = np.array(
            Image.open(
                a_path
            ).convert(
                "RGB"
            )
        )


        # ----------------------------------------------------
        # IMAGE B
        # ----------------------------------------------------

        b = np.array(
            Image.open(
                b_path
            ).convert(
                "RGB"
            )
        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = np.array(
            Image.open(
                label_path
            )
        )


        # ----------------------------------------------------
        # NORMALIZE IMAGES
        # ----------------------------------------------------

        a = (
            a.astype(
                np.float32
            )
            / 255.0
        )


        b = (
            b.astype(
                np.float32
            )
            / 255.0
        )


        # ----------------------------------------------------
        # BINARY LABEL
        # ----------------------------------------------------

        label = (
            label > 127
        ).astype(
            np.float32
        )


        # ----------------------------------------------------
        # HWC → CHW
        # ----------------------------------------------------

        a = torch.from_numpy(
            a.transpose(
                2,
                0,
                1
            )
        )


        b = torch.from_numpy(
            b.transpose(
                2,
                0,
                1
            )
        )


        # ----------------------------------------------------
        # LABEL CHANNEL
        # ----------------------------------------------------

        label = torch.from_numpy(
            label
        ).unsqueeze(
            0
        )


        return (
            a,
            b,
            label,
            a_path.name
        )


# ============================================================
# LOAD DATASET
# ============================================================

dataset = TestDataset()


print()
print("=" * 70)
print("TEST DATA")
print("=" * 70)


print(
    "Complete test pairs:",
    len(dataset)
)


if len(dataset) == 0:

    raise RuntimeError(
        "\nNo complete LEVIR-CD+ test pairs were found.\n\n"
        "Expected structure:\n"
        f"{TEST_ROOT}\\A\n"
        f"{TEST_ROOT}\\B\n"
        f"{TEST_ROOT}\\labels"
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


# ============================================================
# GET STATE DICT
# ============================================================

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
# CREATE MODEL
# ============================================================

model = SiameseChangeDetector()


# ============================================================
# LOAD WEIGHTS
# ============================================================

# Strict loading is intentional.
#
# We want to make absolutely sure that the architecture
# matches the trained checkpoint exactly.

model.load_state_dict(
    state_dict,
    strict=True
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    DEVICE
)


model.eval()


print()
print(
    "Model loaded successfully."
)


# ============================================================
# DATA LOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# METRIC COUNTERS
# ============================================================

tp = 0
tn = 0
fp = 0
fn = 0


processed = 0


# ============================================================
# RUN INFERENCE
# ============================================================

print()
print("=" * 70)
print("RUNNING CHANGE DETECTION")
print("=" * 70)


with torch.no_grad():

    for (
        image_a,
        image_b,
        labels,
        filenames
    ) in loader:


        # ----------------------------------------------------
        # MOVE TO DEVICE
        # ----------------------------------------------------

        image_a = image_a.to(
            DEVICE,
            non_blocking=True
        )


        image_b = image_b.to(
            DEVICE,
            non_blocking=True
        )


        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        # ====================================================
        # MODEL
        # ====================================================

        logits = model(
            image_a,
            image_b
        )


        # ====================================================
        # SIGMOID
        # ====================================================

        probabilities = torch.sigmoid(
            logits
        )


        # ====================================================
        # THRESHOLD
        # ====================================================

        predictions = (
            probabilities > 0.5
        )


        # ====================================================
        # TRUE POSITIVES
        # ====================================================

        tp += int(
            (
                predictions
                & (labels > 0.5)
            )
            .sum()
            .item()
        )


        # ====================================================
        # TRUE NEGATIVES
        # ====================================================

        tn += int(
            (
                (~predictions)
                & (labels <= 0.5)
            )
            .sum()
            .item()
        )


        # ====================================================
        # FALSE POSITIVES
        # ====================================================

        fp += int(
            (
                predictions
                & (labels <= 0.5)
            )
            .sum()
            .item()
        )


        # ====================================================
        # FALSE NEGATIVES
        # ====================================================

        fn += int(
            (
                (~predictions)
                & (labels > 0.5)
            )
            .sum()
            .item()
        )


        # ====================================================
        # SAVE FIRST 20 PREDICTIONS
        # ====================================================

        for i in range(
            len(filenames)
        ):

            if processed < 20:

                prediction = (
                    predictions[i]
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype(
                        np.uint8
                    )
                    * 255
                )


                output_path = (
                    VIS_DIR
                    / filenames[i]
                )


                Image.fromarray(
                    prediction
                ).save(
                    output_path
                )


            processed += 1


        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            processed % 20 == 0
            or processed == len(dataset)
        ):

            print(
                f"Processed: "
                f"{processed}/{len(dataset)}"
            )


# ============================================================
# METRICS
# ============================================================

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0.0
)


recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0.0
)


dice = (
    2 * tp
    / (
        2 * tp
        + fp
        + fn
    )
    if (
        2 * tp
        + fp
        + fn
    ) > 0
    else 0.0
)


iou = (
    tp
    / (
        tp
        + fp
        + fn
    )
    if (
        tp
        + fp
        + fn
    ) > 0
    else 0.0
)


accuracy = (
    (tp + tn)
    / (
        tp
        + tn
        + fp
        + fn
    )
    if (
        tp
        + tn
        + fp
        + fn
    ) > 0
    else 0.0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("CHANGE DETECTION RESULTS")
print("=" * 70)


print(
    f"Test pairs      : {len(dataset)}"
)


print(
    f"Precision       : {precision:.4f}"
)


print(
    f"Recall          : {recall:.4f}"
)


print(
    f"Dice / F1       : {dice:.4f}"
)


print(
    f"IoU             : {iou:.4f}"
)


print(
    f"Pixel Accuracy  : {accuracy:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

final_results = {

    "dataset":
        "LEVIR-CD+",

    "test_pairs":
        len(dataset),

    "precision":
        precision,

    "recall":
        recall,

    "dice_f1":
        dice,

    "iou":
        iou,

    "pixel_accuracy":
        accuracy
}


results_path = (
    OUTPUT_ROOT
    / "change_detection_results.json"
)


with open(
    results_path,
    "w"
) as f:

    json.dump(
        final_results,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("Results saved:")
print(results_path)


print()
print("Visualizations:")
print(VIS_DIR)


print()
print("=" * 70)
print("CHANGE DETECTION COMPLETE")
print("=" * 70)

print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETE")
print("=" * 70)