
from pathlib import Path
import time
import json

import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from train import KuroSiwoDataset


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_DATA_DIR = (
    ROOT / "data" / "external" / "KuroSiwo" / "test_GRD"
)

MODEL_PATH = ROOT / "artifacts" / "flood_unet.pth"

RESULTS_PATH = ROOT / "artifacts" / "evaluation_report.json"


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

BATCH_SIZE = 1

# None = evaluate the complete test dataset.
# Do not set a sample limit for the full evaluation.
MAX_SAMPLES = None

# Print progress every 100 samples.
PROGRESS_INTERVAL = 100


# ============================================================
# LOAD MODEL
# ============================================================

print("\n======================================")
print("       FLOODSENSE EVALUATION")
print("======================================")

print(f"Device: {DEVICE}")
print(f"Test directory: {TEST_DATA_DIR}")
print(f"Checkpoint: {MODEL_PATH}")


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )


if not TEST_DATA_DIR.exists():
    raise FileNotFoundError(
        f"Test dataset directory not found: {TEST_DATA_DIR}"
    )


model = UNet(
    in_channels=7,
    out_channels=1
).to(DEVICE)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nTrained model loaded successfully.")


# ============================================================
# LOAD COMPLETE TEST DATASET
# ============================================================

dataset = KuroSiwoDataset(TEST_DATA_DIR)

total_samples = len(dataset)


if total_samples == 0:
    raise RuntimeError(
        "No test samples were found."
    )


if MAX_SAMPLES is not None:

    sample_count = min(
        MAX_SAMPLES,
        total_samples
    )

    dataset = torch.utils.data.Subset(
        dataset,
        range(sample_count)
    )

else:

    sample_count = total_samples


print("\n======================================")
print("       TEST DATASET INFORMATION")
print("======================================")

print(f"Total test samples: {total_samples}")
print(f"Samples selected: {sample_count}")


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# METRIC COUNTERS
# ============================================================

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0

total_correct = 0
total_valid_pixels = 0

evaluated_samples = 0
skipped_samples = 0


# ============================================================
# EVALUATION
# ============================================================

start_time = time.time()

print("\n======================================")
print("       STARTING FULL EVALUATION")
print("======================================")

print(
    "This evaluation may take several hours "
    "on CPU. Please keep the terminal running.\n"
)


with torch.inference_mode():

    for batch_index, batch in enumerate(
        loader,
        start=1
    ):

        images, labels, valid = batch

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        valid = valid.to(DEVICE).bool()

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = probabilities >= 0.5

        targets = labels >= 0.5


        # ----------------------------------------------------
        # VALID PIXELS
        # ----------------------------------------------------

        valid_pixels = valid.sum().item()


        if valid_pixels == 0:

            skipped_samples += images.shape[0]

        else:

            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            true_positive += (
                predictions
                & targets
                & valid
            ).sum().item()

            false_positive += (
                predictions
                & (~targets)
                & valid
            ).sum().item()

            false_negative += (
                (~predictions)
                & targets
                & valid
            ).sum().item()

            true_negative += (
                (~predictions)
                & (~targets)
                & valid
            ).sum().item()


            # ------------------------------------------------
            # PIXEL ACCURACY
            # ------------------------------------------------

            total_correct += (
                (predictions == targets)
                & valid
            ).sum().item()

            total_valid_pixels += valid_pixels

            evaluated_samples += images.shape[0]


        # ----------------------------------------------------
        # PROGRESS REPORT
        # ----------------------------------------------------

        if (
            batch_index % PROGRESS_INTERVAL == 0
            or batch_index == sample_count
        ):

            elapsed = time.time() - start_time

            average_time = (
                elapsed / batch_index
            )

            remaining_samples = (
                sample_count - batch_index
            )

            estimated_remaining = (
                average_time * remaining_samples
            )

            percentage = (
                batch_index / sample_count
            ) * 100


            print(
                f"Progress: {batch_index}/{sample_count} "
                f"({percentage:.2f}%) "
                f"| Elapsed: {elapsed / 60:.2f} min "
                f"| Avg/sample: {average_time:.2f} sec "
                f"| Estimated remaining: "
                f"{estimated_remaining / 60:.2f} min",
                flush=True
            )


# ============================================================
# CALCULATE METRICS
# ============================================================

# Flood IoU:
# TP / (TP + FP + FN)

iou_denominator = (
    true_positive
    + false_positive
    + false_negative
)


flood_iou = (
    true_positive / iou_denominator
    if iou_denominator > 0
    else 0.0
)


# Pixel accuracy:
# (TP + TN) / all valid pixels

pixel_accuracy = (
    total_correct / total_valid_pixels
    if total_valid_pixels > 0
    else 0.0
)


# Precision:
# TP / (TP + FP)

precision_denominator = (
    true_positive + false_positive
)


precision = (
    true_positive / precision_denominator
    if precision_denominator > 0
    else 0.0
)


# Recall:
# TP / (TP + FN)

recall_denominator = (
    true_positive + false_negative
)


recall = (
    true_positive / recall_denominator
    if recall_denominator > 0
    else 0.0
)


# F1-score:
# 2 * Precision * Recall / (Precision + Recall)

f1_denominator = (
    precision + recall
)


f1_score = (
    2 * precision * recall / f1_denominator
    if f1_denominator > 0
    else 0.0
)


# ============================================================
# EVALUATION DURATION
# ============================================================

total_duration = (
    time.time() - start_time
)


# ============================================================
# RESULTS
# ============================================================

results = {

    "dataset": "Kuro Siwo test_GRD",

    "checkpoint": str(MODEL_PATH),

    "total_test_samples": total_samples,

    "samples_selected": sample_count,

    "evaluated_samples": evaluated_samples,

    "skipped_samples": skipped_samples,

    "batch_size": BATCH_SIZE,

    "threshold": 0.5,

    "pixel_accuracy": pixel_accuracy,

    "flood_iou": flood_iou,

    "precision": precision,

    "recall": recall,

    "f1_score": f1_score,

    "pixel_accuracy_percent": (
        pixel_accuracy * 100
    ),

    "flood_iou_percent": (
        flood_iou * 100
    ),

    "precision_percent": (
        precision * 100
    ),

    "recall_percent": (
        recall * 100
    ),

    "f1_score_percent": (
        f1_score * 100
    ),

    "true_positive": true_positive,

    "false_positive": false_positive,

    "false_negative": false_negative,

    "true_negative": true_negative,

    "total_valid_pixels": total_valid_pixels,

    "elapsed_seconds": total_duration,

    "elapsed_minutes": (
        total_duration / 60
    ),

    "elapsed_hours": (
        total_duration / 3600
    )

}


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print("\n======================================")
print("       EVALUATION COMPLETED")
print("======================================")

print(
    f"Total test samples: {total_samples}"
)

print(
    f"Evaluated samples: {evaluated_samples}"
)

print(
    f"Skipped samples: {skipped_samples}"
)

print(
    f"Pixel accuracy: "
    f"{pixel_accuracy * 100:.2f}%"
)

print(
    f"Flood IoU: "
    f"{flood_iou * 100:.2f}%"
)

print(
    f"Precision: "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall: "
    f"{recall * 100:.2f}%"
)

print(
    f"F1-score: "
    f"{f1_score * 100:.2f}%"
)

print(
    f"Evaluation duration: "
    f"{total_duration / 60:.2f} minutes"
)

print(
    f"Evaluation duration: "
    f"{total_duration / 3600:.2f} hours"
)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


print("\nResults saved to:")

print(RESULTS_PATH)

print("\n======================================")
print("       FLOODSENSE EVALUATION DONE")
print("======================================")
