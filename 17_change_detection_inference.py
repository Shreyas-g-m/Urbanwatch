# ============================================================
# URBANWATCH — LEVIR-CD+ CHANGE DETECTION INFERENCE
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "results"
    / "change_detection"
    / "best_model.pt"
)

TEST_ROOT = (
    ROOT
    / "data"
    / "levir_cd"
    / "test"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "change_detection_inference"
)

VIS_DIR = OUTPUT_ROOT / "visualizations"
PRED_DIR = OUTPUT_ROOT / "predictions"

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

VIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PRED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CROP_SIZE = 512
THRESHOLD = 0.5

# Process the four 512x512 tiles of each 1024x1024 image
TILES = [
    (0, 0),
    (0, 512),
    (512, 0),
    (512, 512),
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("URBANWATCH — LEVIR-CD+ CHANGE DETECTION INFERENCE")
print("=" * 70)

print()
print("Model:")
print(MODEL_PATH)
print("Exists:", MODEL_PATH.exists())

print()
print("Test dataset:")
print(TEST_ROOT)
print("Exists:", TEST_ROOT.exists())

print()
print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print("Inference mode:")
print("Full 1024x1024 image → four 512x512 tiles")


# ============================================================
# PATH CHECKS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nChange detection model not found:\n{MODEL_PATH}"
    )

if not TEST_ROOT.exists():
    raise FileNotFoundError(
        f"\nLEVIR-CD+ test dataset not found:\n{TEST_ROOT}"
    )


# ============================================================
# MODEL
# EXACT ARCHITECTURE USED BY FINAL EVALUATION
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


# ============================================================
# ENCODER
# ============================================================

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


# ============================================================
# DECODER
# ============================================================

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
# EXACT MODEL USED FOR EVALUATION
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

        # Final prediction
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

        # ----------------------------------------------------
        # IMPORTANT:
        # Encode A and B separately using the SAME encoder.
        # ----------------------------------------------------

        a1, a2, a3, a4 = self.encoder(a)

        b1, b2, b3, b4 = self.encoder(b)

        # ----------------------------------------------------
        # Multi-scale absolute feature differences
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Restore original tile size
        # ----------------------------------------------------

        x = F.interpolate(
            x,
            size=a.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return self.final(x)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = SiameseUNet().to(
    DEVICE
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()

print()
print("Model loaded successfully.")

if isinstance(checkpoint, dict):

    if "epoch" in checkpoint:
        print(
            "Checkpoint epoch:",
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
# DATASET DISCOVERY
# ============================================================

A_DIR = TEST_ROOT / "A"
B_DIR = TEST_ROOT / "B"

if not A_DIR.exists():
    raise FileNotFoundError(
        f"A directory not found:\n{A_DIR}"
    )

if not B_DIR.exists():
    raise FileNotFoundError(
        f"B directory not found:\n{B_DIR}"
    )


files = sorted(
    [
        p.name
        for p in A_DIR.glob("*.png")
        if (B_DIR / p.name).exists()
    ]
)

print()
print("=" * 70)
print("TEST DATA")
print("=" * 70)

print(
    "Complete image pairs:",
    len(files)
)

if len(files) == 0:

    raise RuntimeError(
        "No complete A/B image pairs found."
    )


# ============================================================
# HELPER — LOAD IMAGE
# ============================================================

def load_rgb(path):

    image = np.array(
        Image.open(path).convert("RGB"),
        dtype=np.float32
    )

    image /= 255.0

    return image


# ============================================================
# HELPER — IMAGE TO TENSOR
# ============================================================

def image_to_tensor(image):

    tensor = torch.from_numpy(
        image.transpose(2, 0, 1)
    ).float()

    return tensor.unsqueeze(0)


# ============================================================
# INFERENCE ON ONE 512x512 TILE
# ============================================================

@torch.no_grad()
def predict_tile(
    image_a,
    image_b
):

    tensor_a = image_to_tensor(
        image_a
    ).to(
        DEVICE,
        non_blocking=True
    )

    tensor_b = image_to_tensor(
        image_b
    ).to(
        DEVICE,
        non_blocking=True
    )

    # Same inference setup as evaluation
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=torch.cuda.is_available()
    ):

        logits = model(
            tensor_a,
            tensor_b
        )

    probabilities = torch.sigmoid(
        logits
    )

    prediction = (
        probabilities >= THRESHOLD
    ).float()

    prediction = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
    )

    return prediction


# ============================================================
# FULL IMAGE INFERENCE
# ============================================================

def predict_full_image(
    image_a,
    image_b
):

    height, width = image_a.shape[:2]

    if (
        height != 1024
        or width != 1024
    ):

        raise ValueError(
            f"Expected 1024x1024 images, "
            f"got {height}x{width}"
        )

    full_prediction = np.zeros(
        (1024, 1024),
        dtype=np.uint8
    )

    # Four 512x512 tiles
    for top, left in TILES:

        tile_a = image_a[
            top:top + CROP_SIZE,
            left:left + CROP_SIZE
        ]

        tile_b = image_b[
            top:top + CROP_SIZE,
            left:left + CROP_SIZE
        ]

        tile_prediction = predict_tile(
            tile_a,
            tile_b
        )

        full_prediction[
            top:top + CROP_SIZE,
            left:left + CROP_SIZE
        ] = (
            tile_prediction * 255
        ).astype(
            np.uint8
        )

    return full_prediction


# ============================================================
# RUN INFERENCE
# ============================================================

print()
print("=" * 70)
print("RUNNING CHANGE DETECTION")
print("=" * 70)

processed = 0

# Save visualizations for first 10 images
MAX_VISUALIZATIONS = 10


with torch.no_grad():

    for filename in files:

        a_path = A_DIR / filename
        b_path = B_DIR / filename

        # ----------------------------------------------------
        # Load full images
        # ----------------------------------------------------

        image_a = load_rgb(
            a_path
        )

        image_b = load_rgb(
            b_path
        )

        # ----------------------------------------------------
        # Predict complete 1024x1024 image
        # ----------------------------------------------------

        prediction = predict_full_image(
            image_a,
            image_b
        )

        # ----------------------------------------------------
        # Save prediction
        # ----------------------------------------------------

        prediction_path = (
            PRED_DIR / filename
        )

        Image.fromarray(
            prediction
        ).save(
            prediction_path
        )

        # ----------------------------------------------------
        # Save visualizations
        # ----------------------------------------------------

        if processed < MAX_VISUALIZATIONS:

            image_a_uint8 = (
                image_a * 255
            ).clip(
                0,
                255
            ).astype(
                np.uint8
            )

            image_b_uint8 = (
                image_b * 255
            ).clip(
                0,
                255
            ).astype(
                np.uint8
            )

            pred_rgb = (
                Image.fromarray(
                    prediction
                ).convert("RGB")
            )

            # 4-panel visualization
            canvas = Image.new(
                "RGB",
                (
                    1024 * 3,
                    1024
                )
            )

            canvas.paste(
                Image.fromarray(
                    image_a_uint8
                ),
                (0, 0)
            )

            canvas.paste(
                Image.fromarray(
                    image_b_uint8
                ),
                (1024, 0)
            )

            canvas.paste(
                pred_rgb,
                (2048, 0)
            )

            visualization_path = (
                VIS_DIR
                / f"comparison_{processed + 1:02d}.png"
            )

            canvas.save(
                visualization_path
            )

        processed += 1

        if (
            processed == 1
            or processed % 20 == 0
            or processed == len(files)
        ):

            print(
                f"Processed: "
                f"{processed}/{len(files)}"
            )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "dataset":
        "LEVIR-CD+",

    "images_processed":
        len(files),

    "input_size":
        [1024, 1024],

    "inference_tile_size":
        [512, 512],

    "tiles_per_image":
        4,

    "threshold":
        THRESHOLD,

    "model":
        str(MODEL_PATH),

    "predictions":
        str(PRED_DIR),

    "visualizations":
        str(VIS_DIR)
}


summary_path = (
    OUTPUT_ROOT
    / "inference_summary.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("CHANGE DETECTION INFERENCE COMPLETE")
print("=" * 70)

print()
print("Images processed:")
print(len(files))

print()
print("Predictions:")
print(PRED_DIR)

print()
print("Visualizations:")
print(VIS_DIR)

print()
print("Summary:")
print(summary_path)

print()
print("=" * 70)