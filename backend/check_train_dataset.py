
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "external" / "KuroSiwo" / "train_GRD"

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

archives = sorted(TRAIN_DIR.glob("*.tar"))

print("Training archives:", len(archives))

if not archives:
    raise SystemExit("ERROR: No training TAR archives found.")

total_samples = 0
complete_samples = 0
incomplete_samples = 0

for archive in archives:
    print(f"\nChecking: {archive.name}")

    sample_files = {}

    with tarfile.open(archive, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue

            name = Path(member.name).name

            if name.endswith(".info.json"):
                sample_id = name.removesuffix(".info.json")
                sample_files.setdefault(sample_id, set())

            for filename in REQUIRED_FILES:
                if name.endswith("." + filename):
                    sample_id = name.removesuffix("." + filename)
                    sample_files.setdefault(sample_id, set()).add(filename)

    archive_samples = [
        sample_id
        for sample_id, files in sample_files.items()
        if sample_id.isdigit()
    ]

    archive_complete = 0
    archive_incomplete = 0

    for sample_id in archive_samples:
        total_samples += 1

        files = sample_files[sample_id]
        missing = set(REQUIRED_FILES) - files

        if missing:
            incomplete_samples += 1
            archive_incomplete += 1

            print(
                f"Sample {sample_id}: "
                f"MISSING {', '.join(sorted(missing))}"
            )
        else:
            complete_samples += 1
            archive_complete += 1

    print("Samples:", len(archive_samples))
    print("Complete:", archive_complete)
    print("Incomplete:", archive_incomplete)

print("\n========== TRAINING DATASET SUMMARY ==========")
print("Total samples:", total_samples)
print("Complete samples:", complete_samples)
print("Incomplete samples:", incomplete_samples)
print("============================================")
