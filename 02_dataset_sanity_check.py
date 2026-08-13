from pathlib import Path
from PIL import Image
import numpy as np
import json

ROOT = Path(__file__).resolve().parent

print("=" * 70)
print("URBANWATCH DATASET SANITY CHECK")
print("=" * 70)

# ============================================================
# 1. LEVIR-CD+
# ============================================================

print("\n" + "=" * 70)
print("1. LEVIR-CD+")
print("=" * 70)

levir = ROOT / "LEVIR-CD+"

a = levir / "train" / "A" / "train_1.png"
b = levir / "train" / "B" / "train_1.png"
label = levir / "train" / "label" / "train_1.png"

for name, path in [
    ("Image A", a),
    ("Image B", b),
    ("Change Label", label),
]:
    print(f"\n{name}")
    print("Path:", path)
    print("Exists:", path.exists())

    if path.exists():
        with Image.open(path) as img:
            arr = np.array(img)

        print("Shape:", arr.shape)
        print("Data type:", arr.dtype)
        print("Min:", arr.min())
        print("Max:", arr.max())

        if name == "Change Label":
            print("Unique values:", np.unique(arr))


# ============================================================
# 2. LoveDA
# ============================================================

print("\n" + "=" * 70)
print("2. LoveDA")
print("=" * 70)

loveda_image = ROOT / "Train" / "Rural" / "images_png" / "0.png"
loveda_mask = ROOT / "Train" / "Rural" / "masks_png" / "0.png"

for name, path in [
    ("Image", loveda_image),
    ("Mask", loveda_mask),
]:
    print(f"\n{name}")
    print("Path:", path)
    print("Exists:", path.exists())

    if path.exists():
        with Image.open(path) as img:
            arr = np.array(img)

        print("Shape:", arr.shape)
        print("Data type:", arr.dtype)
        print("Min:", arr.min())
        print("Max:", arr.max())
        print("Unique values:", np.unique(arr)[:30])


# ============================================================
# 3. SpaceNet 2
# ============================================================

print("\n" + "=" * 70)
print("3. SpaceNet 2")
print("=" * 70)

spacenet = ROOT / "AOI_3_Paris_Train"

# RGB-PanSharpen image
rgb_folder = spacenet / "RGB-PanSharpen"

tif_files = sorted(rgb_folder.glob("*.tif"))

print("\nRGB-PanSharpen TIFF count:", len(tif_files))

if tif_files:
    image_path = tif_files[0]

    print("\nSample TIFF:")
    print("Path:", image_path)
    print("Size:", image_path.stat().st_size / (1024 ** 2), "MB")

# Matching GeoJSON
geojson_folder = spacenet / "geojson" / "buildings"

geojson_files = sorted(geojson_folder.glob("*.geojson"))

print("\nBuilding GeoJSON count:", len(geojson_files))

if geojson_files:
    geojson_path = geojson_files[0]

    print("\nSample GeoJSON:")
    print("Path:", geojson_path)

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Type:", data.get("type"))
    print("CRS:", data.get("crs"))
    print("Number of features:", len(data.get("features", [])))

    if data.get("features"):
        feature = data["features"][0]

        print("\nFirst feature:")
        print("Geometry type:", feature["geometry"]["type"])
        print("Properties:", feature["properties"])


# ============================================================
# 4. Summary
# ============================================================

print("\n" + "=" * 70)
print("SANITY CHECK COMPLETE")
print("=" * 70)