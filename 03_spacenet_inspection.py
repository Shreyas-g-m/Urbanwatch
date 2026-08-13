from pathlib import Path
import json
import tifffile

ROOT = Path(__file__).resolve().parent

SPACENET = ROOT / "AOI_3_Paris_Train"

IMAGE_DIR = SPACENET / "RGB-PanSharpen"
LABEL_DIR = SPACENET / "geojson" / "buildings"

print("=" * 70)
print("SPACENET 2 INSPECTION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Inspect TIFF files
# ------------------------------------------------------------

tif_files = sorted(IMAGE_DIR.glob("*.tif"))

print("\nNumber of RGB-PanSharpen images:", len(tif_files))

print("\nInspecting first 3 TIFFs:")

for path in tif_files[:3]:

    print("\n", path.name)

    image = tifffile.imread(path)

    print("Shape:", image.shape)
    print("Data type:", image.dtype)
    print("Min:", image.min())
    print("Max:", image.max())


# ------------------------------------------------------------
# 2. Find GeoJSON files containing buildings
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SEARCHING FOR NON-EMPTY BUILDING LABELS")
print("=" * 70)

non_empty = []

for path in sorted(LABEL_DIR.glob("*.geojson")):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])

    if len(features) > 0:
        non_empty.append((path, len(features)))

print("\nGeoJSON files containing buildings:", len(non_empty))

print("\nFirst 10 non-empty examples:")

for path, count in non_empty[:10]:

    print(f"{path.name:70} buildings: {count}")


# ------------------------------------------------------------
# 3. Match image and label
# ------------------------------------------------------------

if non_empty:

    label_path, count = non_empty[0]

    image_name = label_path.name.replace(
        "buildings_", "RGB-PanSharpen_"
    ).replace(
        ".geojson", ".tif"
    )

    image_path = IMAGE_DIR / image_name

    print("\n" + "=" * 70)
    print("MATCHED TRAINING SAMPLE")
    print("=" * 70)

    print("Image:")
    print(image_path)

    print("\nLabel:")
    print(label_path)

    print("\nBuilding count:", count)

    print("\nImage exists:", image_path.exists())
    print("Label exists:", label_path.exists())

else:

    print("\nNo non-empty GeoJSON files found.")