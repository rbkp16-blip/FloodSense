
from pathlib import Path
import argparse
import io
import shutil
import tarfile

import numpy as np
import rasterio
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = ROOT / "data" / "external" / "KuroSiwo" / "test_GRD"
TIFF_DIR = ROOT / "data" / "test_inputs"
JPG_DIR = ROOT / "data" / "jpg_previews"

BAND_NAMES = [
    "flood_vv.npy",
    "flood_vh.npy",
    "sec1_vv.npy",
    "sec1_vh.npy",
    "sec2_vv.npy",
    "sec2_vh.npy",
    "dem.npy",
]


def load_array(tar, member):
    extracted = tar.extractfile(member)

    if extracted is None:
        raise ValueError(f"Cannot read {member.name}")

    array = np.load(
        io.BytesIO(extracted.read()),
        allow_pickle=False,
    )

    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(f"Invalid shape: {array.shape}")

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Non-numeric band")

    array = array.astype(np.float32)

    if not np.isfinite(array).all():
        raise ValueError("Non-finite pixel values")

    return array


def make_preview(band):
    """Create an 8-bit visualization of a SAR band."""
    values = band[np.isfinite(band)]

    if values.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)

    low, high = np.percentile(values, [2, 98])

    if high <= low:
        return np.zeros(band.shape, dtype=np.uint8)

    scaled = (band - low) / (high - low)
    scaled = np.clip(scaled, 0, 1)

    return (scaled * 255).astype(np.uint8)


def save_pair(image, sample_name):
    tiff_path = TIFF_DIR / f"{sample_name}.tif"
    jpg_path = JPG_DIR / f"{sample_name}.jpg"

    if tiff_path.exists() and jpg_path.exists():
        return "existing"

    # Write to temporary paths first.
    temp_tiff = TIFF_DIR / f"{sample_name}.partial.tif"
    temp_jpg = JPG_DIR / f"{sample_name}.partial.jpg"

    try:
        with rasterio.open(
            temp_tiff,
            "w",
            driver="GTiff",
            height=image.shape[1],
            width=image.shape[2],
            count=7,
            dtype="float32",
            compress="deflate",
        ) as dst:
            dst.write(image)

            for index, band_name in enumerate(
                BAND_NAMES, start=1
            ):
                dst.set_band_description(
                    index,
                    band_name.removesuffix(".npy"),
                )

        # Display the first VV band as a grayscale SAR preview.
        preview = make_preview(image[0])

        Image.fromarray(preview).save(
            temp_jpg,
            format="JPEG",
            quality=92,
        )

        # Validate both outputs before publishing them.
        with rasterio.open(temp_tiff) as src:
            if (
                src.count != 7
                or src.width != image.shape[2]
                or src.height != image.shape[1]
            ):
                raise ValueError("TIFF validation failed")

        with Image.open(temp_jpg) as jpg:
            jpg.verify()

        temp_tiff.replace(tiff_path)
        temp_jpg.replace(jpg_path)

        return "created"

    finally:
        temp_tiff.unlink(missing_ok=True)
        temp_jpg.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of complete output pairs",
    )

    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=40,
        help="Minimum free disk space to preserve",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")

    if args.min_free_gb < 0:
        parser.error("--min-free-gb cannot be negative")

    archives = sorted(SOURCE_DIR.glob("*.tar"))

    if not archives:
        raise FileNotFoundError(
            f"No TAR archives found in {SOURCE_DIR}"
        )

    TIFF_DIR.mkdir(parents=True, exist_ok=True)
    JPG_DIR.mkdir(parents=True, exist_ok=True)

    created = 0
    existing = 0
    skipped = 0
    processed = 0

    print("=" * 60)
    print("FLOODSENSE - KURO SIWO BULK CONVERTER")
    print("=" * 60)
    print(f"Archives found: {len(archives)}")
    print(f"Target pairs: {args.limit}")
    print(f"TIFF directory: {TIFF_DIR}")
    print(f"JPG directory: {JPG_DIR}")

    for archive_index, archive_path in enumerate(archives):
        print(f"\nReading {archive_path.name}")

        with tarfile.open(archive_path, "r") as tar:
            members = {
                Path(member.name).name: member
                for member in tar.getmembers()
                if member.isfile()
            }

            sample_ids = sorted(
                name.removesuffix(".info.json")
                for name in members
                if name.endswith(".info.json")
            )

            for sample_id in sample_ids:
                if processed >= args.limit:
                    break

                free_gb = (
                    shutil.disk_usage(TIFF_DIR).free
                    / (1024 ** 3)
                )

                if free_gb < args.min_free_gb:
                    print(
                        f"\nStopping: only {free_gb:.2f} GB free."
                    )
                    return

                sample_name = (
                    f"kurosiwo_test_{archive_index:02d}_{sample_id}"
                )

                tiff_path = TIFF_DIR / f"{sample_name}.tif"
                jpg_path = JPG_DIR / f"{sample_name}.jpg"

                if tiff_path.exists() and jpg_path.exists():
                    existing += 1
                    processed += 1
                    continue

                required = [
                    f"{sample_id}.{band_name}"
                    for band_name in BAND_NAMES
                ]

                if not all(name in members for name in required):
                    skipped += 1
                    continue

                try:
                    bands = [
                        load_array(tar, members[name])
                        for name in required
                    ]

                    shape = bands[0].shape

                    if not all(
                        band.shape == shape for band in bands
                    ):
                        raise ValueError("Band shape mismatch")

                    image = np.stack(
                        bands, axis=0
                    ).astype(np.float32)

                    result = save_pair(image, sample_name)

                    if result == "created":
                        created += 1
                    else:
                        existing += 1

                    processed += 1

                    if processed % 100 == 0:
                        print(
                            f"Completed: {processed}/{args.limit} "
                            f"| New: {created} "
                            f"| Existing: {existing} "
                            f"| Skipped: {skipped}",
                            flush=True,
                        )

                except Exception as exc:
                    skipped += 1
                    print(
                        f"Skipped {sample_name}: {exc}"
                    )

        if processed >= args.limit:
            break

    print("\n" + "=" * 60)
    print("CONVERSION FINISHED")
    print("=" * 60)
    print(f"Completed pairs: {processed}")
    print(f"New pairs: {created}")
    print(f"Existing pairs: {existing}")
    print(f"Skipped samples: {skipped}")


if __name__ == "__main__":
    main()
