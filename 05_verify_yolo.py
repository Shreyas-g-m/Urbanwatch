from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "data" / "yolo_buildings"

IMAGE_DIR = DATASET / "images" / "train"
LABEL_DIR = DATASET / "labels" / "train"

OUTPUT_DIR = ROOT / "results" / "yolo_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 10


def load_yolo_labels(label_path):
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                continue

            class_id, xc, yc, w, h = map(float, values)

            boxes.append(
                (
                    int(class_id),
                    xc,
                    yc,
                    w,
                    h,
                )
            )

    return boxes


images = sorted(IMAGE_DIR.glob("*.png"))

if not images:
    raise RuntimeError("No YOLO training images found.")

random.seed(42)

# Prefer images that actually contain buildings
candidates = []

for image_path in images:

    label_path = LABEL_DIR / (
        image_path.stem + ".txt"
    )

    boxes = load_yolo_labels(label_path)

    if boxes:
        candidates.append(image_path)

print("Images with building annotations:", len(candidates))

samples = random.sample(
    candidates,
    min(NUM_SAMPLES, len(candidates))
)


for index, image_path in enumerate(samples, start=1):

    label_path = LABEL_DIR / (
        image_path.stem + ".txt"
    )

    image = Image.open(image_path).convert("RGB")

    draw = ImageDraw.Draw(image)

    width, height = image.size

    boxes = load_yolo_labels(label_path)

    for class_id, xc, yc, bw, bh in boxes:

        x_center = xc * width
        y_center = yc * height

        box_width = bw * width
        box_height = bh * height

        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2

        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=2,
        )

    output_path = (
        OUTPUT_DIR /
        f"sample_{index}_{image_path.stem}.jpg"
    )

    image.save(output_path)

    print(
        f"[{index}/{len(samples)}] "
        f"{image_path.name} | "
        f"boxes: {len(boxes)}"
    )

print("\nSaved visualizations to:")
print(OUTPUT_DIR)