import tarfile
from pathlib import Path

TEST_DIR = Path("../data/external/KuroSiwo/test_GRD")
archives = sorted(TEST_DIR.glob("*.tar"))

print("Testing archives:", len(archives))
print("Scanning testing dataset...")

total_samples = 0

for archive in archives:
    count = 0

    with tarfile.open(archive, "r|") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".info.json"):
                count += 1

    total_samples += count
    print(f"{archive.name}: {count:,} samples")

print("\nTOTAL TESTING SAMPLES:", f"{total_samples:,}")
