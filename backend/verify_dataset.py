
from pathlib import Path
import rasterio

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "raw" / "S1Hand"
LABEL_DIR = ROOT / "data" / "raw" / "LabelHand"

images = sorted(IMAGE_DIR.glob("*.tif"))
labels = sorted(LABEL_DIR.glob("*.tif"))

def sample_id(path, suffix):
    return path.stem.removesuffix(suffix)

image_map = {sample_id(p, "_S1Hand"): p for p in images}
label_map = {sample_id(p, "_LabelHand"): p for p in labels}
common_ids = sorted(image_map.keys() & label_map.keys())

print(f"Satellite images: {len(images)}")
print(f"Flood labels: {len(labels)}")
print(f"Matching pairs: {len(common_ids)}")

if not common_ids:
    raise SystemExit("ERROR: No matching image-label pairs found.")

for sample in common_ids[:3]:
    with rasterio.open(image_map[sample]) as src:
        print(f"\nImage: {image_map[sample].name}")
        print(f"  Shape: {src.height} x {src.width}")
        print(f"  Bands: {src.count}")
        print(f"  Data type: {src.dtypes}")

    with rasterio.open(label_map[sample]) as src:
        print(f"Label: {label_map[sample].name}")
        print(f"  Shape: {src.height} x {src.width}")
        print(f"  Bands: {src.count}")
        print(f"  Data type: {src.dtypes}")

print("\nDataset file verification completed.")
