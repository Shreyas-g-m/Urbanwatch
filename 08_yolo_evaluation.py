import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import json
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "results"
    / "yolo_training"
    / "spacenet_building_detector-4"
    / "weights"
    / "best.pt"
)

DATASET_YAML = (
    ROOT
    / "data"
    / "yolo_buildings"
    / "dataset.yaml"
)

RESULTS_DIR = ROOT / "results" / "yolo_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("URBANWATCH — YOLO EVALUATION")
print("=" * 70)

print("\nModel:", MODEL_PATH)
print("Exists:", MODEL_PATH.exists())

print("Dataset:", DATASET_YAML)
print("Exists:", DATASET_YAML.exists())

print("\nPyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))

print("\nModel loaded successfully.")


# ============================================================
# VALIDATION
# ============================================================

metrics = model.val(
    data=str(DATASET_YAML),
    split="val",
    imgsz=640,
    batch=4,
    device=0,
    workers=0,
    plots=True,
    project=str(RESULTS_DIR),
    name="validation",
)


# ============================================================
# RESULTS
# ============================================================

precision = float(metrics.box.mp)
recall = float(metrics.box.mr)
map50 = float(metrics.box.map50)
map5095 = float(metrics.box.map)


print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"mAP@50    : {map50:.4f}")
print(f"mAP@50-95 : {map5095:.4f}")


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "model": str(MODEL_PATH),
    "precision": precision,
    "recall": recall,
    "mAP50": map50,
    "mAP50_95": map5095,
}

summary_path = RESULTS_DIR / "evaluation_summary.json"

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)

print("\nSummary saved to:")
print(summary_path)

print("\nEvaluation complete.")