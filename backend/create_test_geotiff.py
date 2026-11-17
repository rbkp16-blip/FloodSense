
from pathlib import Path
import io
import tarfile

import numpy as np
import rasterio


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "data" / "external" / "KuroSiwo" / "test_GRD"

OUTPUT_DIR = ROOT / "data" / "test_inputs"

OUTPUT_PATH = OUTPUT_DIR / "kurosiwo_test_7band.tif"


# ============================================================
# SEVEN INPUT BANDS - SAME ORDER AS train.py
# ============================================================

BAND_NAMES = [
    "flood_vv.npy",
    "flood_vh.npy",
    "sec1_vv.npy",
    "sec1_vh.npy",
    "sec2_vv.npy",
    "sec2_vh.npy",
    "dem.npy",
]


# ============================================================
# LOAD NUMPY ARRAY FROM TAR
# ============================================================

def load_array(tar, member_name):

    extracted = tar.extractfile(member_name)

    if extracted is None:
        raise RuntimeError(
            f"Could not extract: {member_name}"
        )

    array = np.load(
        io.BytesIO(extracted.read()),
        allow_pickle=False
    )

    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, got {array.shape}"
        )

    return array.astype(np.float32)


# ============================================================
# CREATE TEST GEOTIFF
# ============================================================

def main():

    print("\n======================================")
    print("    FLOODSENSE TEST GEOTIFF CREATOR")
    print("======================================")

    archives = sorted(TEST_DIR.glob("*.tar"))

    if not archives:
        raise FileNotFoundError(
            f"No TAR archives found in {TEST_DIR}"
        )

    archive_path = archives[0]

    print(f"\nUsing archive: {archive_path.name}")

    with tarfile.open(archive_path, "r") as tar:

        members = {
            Path(member.name).name: member.name
            for member in tar.getmembers()
            if member.isfile()
        }

        sample_ids = sorted(
            name.removesuffix(".info.json")
            for name in members
            if name.endswith(".info.json")
        )

        if not sample_ids:
            raise RuntimeError(
                "No samples found in the selected archive."
            )

        selected_id = None
        image = None

        for sample_id in sample_ids:

            required_names = [
                f"{sample_id}.{band_name}"
                for band_name in BAND_NAMES
            ]

            if not all(
                name in members
                for name in required_names
            ):
                continue

            bands = [
                load_array(tar, members[name])
                for name in required_names
            ]

            shape = bands[0].shape

            if not all(
                band.shape == shape
                for band in bands
            ):
                continue

            image = np.stack(
                bands,
                axis=0
            ).astype(np.float32)

            selected_id = sample_id
            break

    if image is None:
        raise RuntimeError(
            "Could not find a complete seven-band sample."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    channels, height, width = image.shape

    # The source arrays do not establish a verified
    # geographic transform or CRS. This GeoTIFF is
    # for local model testing, not geographic mapping.

    with rasterio.open(
        OUTPUT_PATH,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=channels,
        dtype="float32",
        compress="deflate",
    ) as dst:

        dst.write(image)

        for index, name in enumerate(
            BAND_NAMES,
            start=1
        ):
            dst.set_band_description(
                index,
                name.removesuffix(".npy")
            )

    print("\n======================================")
    print("       GEOTIFF CREATED")
    print("======================================")

    print(f"Sample ID: {selected_id}")
    print(f"Image shape: {image.shape}")
    print(f"Number of bands: {channels}")
    print(f"Output file: {OUTPUT_PATH}")

    print("\nBand order:")

    for index, name in enumerate(
        BAND_NAMES,
        start=1
    ):
        print(f"{index}. {name}")

    print("\nTest file created successfully.")


if __name__ == "__main__":
    main()
