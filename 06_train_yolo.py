import os

# Fix OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from ultralytics import YOLO
import torch


ROOT = Path(__file__).resolve().parent

DATASET_YAML = ROOT / "data" / "yolo_buildings" / "dataset.yaml"

PROJECT_DIR = ROOT / "results" / "yolo_training"

PROJECT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("URBANWATCH — YOLO BUILDING DETECTION")
print("=" * 70)

print("\nPyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nDataset:")
print(DATASET_YAML)


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

model = YOLO("yolov8m.pt")


# ============================================================
# TRAIN
# ============================================================

if __name__ == "__main__":

    results = model.train(
        data=str(DATASET_YAML),

        # Model
        epochs=100,
        imgsz=640,

        # Your RTX 4070 Laptop GPU
        device=0,

        # Batch size
        batch=8,

        # Windows-safe multiprocessing
        workers=0,

        # Output
        project=str(PROJECT_DIR),
        name="spacenet_building_detector",

        # Reproducibility
        seed=42,

        # Save checkpoints
        save=True,
        save_period=10,

        # Early stopping
        patience=20,

        # Performance
        amp=True,

        # Do not cache images in RAM
        cache=False,

        verbose=True,
    )


    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("\nResults saved to:")
    print(
        PROJECT_DIR /
        "spacenet_building_detector"
    )

    print("\nBest model should be:")
    print(
        PROJECT_DIR /
        "spacenet_building_detector" /
        "weights" /
        "best.pt"
    )