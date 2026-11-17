
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "external" / "KuroSiwo" / "test_GRD"

REQUIRED_FILES = [
    "flood_vv.npy",
    "flood_vh.npy",
    "sec1_vv.npy",
    "sec1_vh.npy",
    "sec2_vv.npy",
    "sec2_vh.npy",
    "dem.npy",
    "mask.npy",
    "valid_mask.npy",
]

archives = sorted(TEST_DIR.glob("*.tar"))

print("Testing archives:", len(archives))

if not archives:
    raise SystemExit("ERROR: No test TAR archives found.")

total_samples = 0
complete_samples = 0
incomplete_samples = 0

for archive in archives:
    print(f"\nChecking: {archive.name}")

    with tarfile.open(archive, "r") as tar:
        sample_files = {}

        for member in tar:
            if not member.isfile():
                continue

            filename = Path(member.name).name

            for required in REQUIRED_FILES:
                suffix = "." + required

                if filename.endswith(suffix):
                    sample_id = filename.removesuffix(suffix)
                    sample_files.setdefault(sample_id, set()).add(required)

        archive_samples = len(sample_files)
        archive_complete = 0
        archive_incomplete = 0

        for sample_id, files in sample_files.items():
            missing = set(REQUIRED_FILES) - files

            if missing:
                archive_incomplete += 1
            else:
                archive_complete += 1

        total_samples += archive_samples
        complete_samples += archive_complete
        incomplete_samples += archive_incomplete

        print("Samples:", archive_samples)
        print("Complete:", archive_complete)
        print("Incomplete:", archive_incomplete)

print("\n========== TEST DATASET SUMMARY ==========")
print("Total samples:", total_samples)
print("Complete samples:", complete_samples)
print("Incomplete samples:", incomplete_samples)
print("========================================")
