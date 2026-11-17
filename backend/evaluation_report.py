
from pathlib import Path
import json

REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "evaluation_report.json"

report = {
    "project": "FloodSense",
    "model": "U-Net",
    "dataset": "SEN1Floods11",
    "training_samples": 16,
    "validation_samples": 4,
    "epochs": 5,
    "pixel_accuracy_percent": 85.07,
    "flood_iou_percent": 38.57,
    "limitations": [
        "The model was trained on only 20 samples.",
        "The validation set contains only 4 samples.",
        "Results are preliminary and are not suitable for real-world flood warnings."
    ]
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=4), encoding="utf-8")

print(f"Evaluation report saved to: {REPORT_PATH}")
