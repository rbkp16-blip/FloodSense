
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.models.unet import UNet


# ============================================================
# FLOODSENSE CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "flood_unet.pth"
EVALUATION_PATH = PROJECT_ROOT / "docs" / "evaluation_report.json"

DEVICE = torch.device("cpu")
INPUT_SIZE = 128
FLOOD_THRESHOLD = 0.5
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(
    title="FloodSense API",
    description=(
        "Flood extent mapping, preliminary risk assessment, "
        "and response prioritization research prototype."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_loaded = False
model_message = "Model has not been loaded."


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """
    Load the trained U-Net checkpoint if it exists.

    The model expects two input channels. If no checkpoint exists,
    the API remains available, but predictions must be treated as
    untrained prototype output.
    """
    global model, model_loaded, model_message

    try:
        model = UNet(in_channels=2, out_channels=1).to(DEVICE)

        if not MODEL_PATH.exists():
            model_loaded = False
            model_message = (
                "No trained checkpoint found. "
                "Place a trained checkpoint at artifacts/flood_unet.pth."
            )
            return

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "model_state_dict",
                checkpoint.get("state_dict", checkpoint),
            )
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()

        model_loaded = True
        model_message = "Trained U-Net checkpoint loaded successfully."

    except Exception as exc:
        model = None
        model_loaded = False
        model_message = f"Model loading failed: {exc}"


load_model()


# ============================================================
# GENERAL ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "project": "FloodSense",
        "status": "running",
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "project": "FloodSense",
        "status": "online",
        "api": "healthy",
        "model_loaded": model_loaded,
        "model_status": (
            "trained_checkpoint_loaded"
            if model_loaded
            else "trained_checkpoint_unavailable"
        ),
        "model_message": model_message,
        "device": str(DEVICE),
    }


@app.get("/evaluation")
def evaluation():
    """
    Return the project's reported evaluation values.

    Values below are prototype defaults. Replace them with values
    measured from a real evaluation run before reporting them as
    experimental results.
    """
    if EVALUATION_PATH.exists():
        try:
            import json

            with EVALUATION_PATH.open("r", encoding="utf-8") as file:
                report = json.load(file)

            if isinstance(report, dict):
                return report

        except Exception:
            pass

    return {
        "project": "FloodSense",
        "model": "U-Net",
        "dataset": "SEN1Floods11",
        "training_samples": 16,
        "validation_samples": 4,
        "epochs": 5,
        "pixel_accuracy_percent": None,
        "flood_iou_percent": None,
        "evaluation_status": "report_not_available",
        "limitations": [
            "Evaluation metrics have not been loaded from a verified report.",
            "The model's performance on unseen flood events is unknown.",
            "Do not use this prototype to issue real-world flood warnings.",
        ],
    }


# ============================================================
# FLOOD PREDICTION
# ============================================================

def normalize_band(band, valid_mask):
    """Percentile-normalize one SAR band using valid pixels."""
    output = np.zeros(band.shape, dtype=np.float32)

    values = band[valid_mask]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return output

    low, high = np.percentile(values, [2, 98])

    if high <= low:
        output[valid_mask] = 0.0
        return output

    normalized = (band - low) / (high - low)
    output[valid_mask] = np.clip(
        normalized[valid_mask], 0.0, 1.0
    )

    return output


def resize_to_model_input(array):
    """Resize a two-band array to the model's 128 x 128 input."""
    tensor = torch.from_numpy(array).float().unsqueeze(0)

    resized = nn.functional.interpolate(
        tensor,
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return resized


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a two-band SAR GeoTIFF and return a preliminary flood mask.

    The input is resized to 128 x 128 for inference. The returned
    probability map therefore represents the model's inference grid,
    not necessarily the original image's full spatial resolution.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please select a GeoTIFF file.",
        )

    if Path(file.filename).suffix.lower() not in {".tif", ".tiff"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a .tif or .tiff file.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 50 MB upload limit.",
        )

    import io

    try:
        with rasterio.MemoryFile(contents) as memory_file:
            with memory_file.open() as dataset:
                if dataset.count < 2:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "The GeoTIFF must contain at least two bands "
                            "(for example, Sentinel-1 VV and VH)."
                        ),
                    )

                width = dataset.width
                height = dataset.height

                # Read the first two bands as floating-point arrays.
                image = dataset.read(
                    indexes=[1, 2],
                    out_dtype="float32",
                )

                band_masks = dataset.read_masks(indexes=[1, 2])
                valid_mask = np.all(band_masks > 0, axis=0)
                valid_mask &= np.all(np.isfinite(image), axis=0)

                # Exclude pixels where either input band is invalid.
                if not np.any(valid_mask):
                    raise HTTPException(
                        status_code=400,
                        detail="No valid pixels were found in this GeoTIFF.",
                    )

                band_1 = normalize_band(image[0], valid_mask)
                band_2 = normalize_band(image[1], valid_mask)

                normalized_image = np.stack(
                    [band_1, band_2], axis=0
                ).astype(np.float32)

                original_valid_pixel_count = int(valid_mask.sum())

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the GeoTIFF: {exc}",
        ) from exc

    # Resize to the spatial dimensions used by this U-Net.
    input_tensor = resize_to_model_input(normalized_image)

    # If no trained checkpoint is available, do not present random
    # neural-network weights as a legitimate flood prediction.
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "A trained U-Net checkpoint is required for prediction. "
                "The API is running, but artifacts/flood_unet.pth "
                "is missing or could not be loaded."
            ),
        )

    try:
        model.eval()

        with torch.no_grad():
            logits = model(input_tensor.to(DEVICE))
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

        probabilities = np.nan_to_num(
            probabilities,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        probabilities = np.clip(probabilities, 0.0, 1.0)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {exc}",
        ) from exc

    flood_mask = probabilities >= FLOOD_THRESHOLD
    predicted_flood_pixel_count = int(flood_mask.sum())
    inference_pixel_count = int(probabilities.size)

    flood_percentage = (
        predicted_flood_pixel_count / inference_pixel_count * 100.0
        if inference_pixel_count
        else 0.0
    )

    return {
        "project": "FloodSense",
        "status": "success",
        "filename": file.filename,
        "width": width,
        "height": height,
        "input_bands": 2,
        "inference_width": INPUT_SIZE,
        "inference_height": INPUT_SIZE,
        "threshold": FLOOD_THRESHOLD,
        "valid_pixel_count": original_valid_pixel_count,
        "predicted_flood_pixel_count": predicted_flood_pixel_count,
        "predicted_flood_pixel_percentage": round(
            flood_percentage, 2
        ),
        "probability_map": probabilities.round(4).tolist(),
        "model": "U-Net",
        "model_checkpoint_loaded": True,
        "warning": (
            "Research prototype only. Predictions depend on the trained "
            "checkpoint and have not been validated for operational "
            "flood warnings."
        ),
    }


# ============================================================
# RESPONSE PRIORITIZATION SCHEMA
# ============================================================

RainfallLevel = Literal[
    "unknown",
    "low",
    "moderate",
    "high",
    "extreme",
]

PopulationExposure = Literal[
    "unknown",
    "low",
    "moderate",
    "high",
    "very_high",
]


class ResponsePlanRequest(BaseModel):
    predicted_flood_pixel_percentage: float = Field(
        ge=0,
        le=100,
        description="Predicted flood share, expressed as a percentage.",
    )
    rainfall_level: RainfallLevel = "unknown"
    population_exposure: PopulationExposure = "unknown"
    critical_infrastructure_exposed: bool = False


# ============================================================
# RESPONSE PRIORITIZATION LOGIC
# ============================================================

def calculate_response_plan(request: ResponsePlanRequest):
    """
    Calculate an explainable prototype risk score from the supplied
    flood share and user-provided exposure indicators.

    This is a heuristic decision-support score, not a calibrated
    probability of harm or an official emergency classification.
    """
    flood_share = request.predicted_flood_pixel_percentage
    rainfall = request.rainfall_level
    population = request.population_exposure
    infrastructure = request.critical_infrastructure_exposed

    score = 0
    reasons = []
    data_gaps = []

    # Flood extent contribution: 0 to 40 points.
    if flood_share < 5:
        score += 5
        reasons.append("Low predicted flood extent.")
    elif flood_share < 15:
        score += 15
        reasons.append("Moderate predicted flood extent.")
    elif flood_share < 30:
        score += 25
        reasons.append("High predicted flood extent.")
    else:
        score += 40
        reasons.append("Very high predicted flood extent.")

    # Rainfall contribution: up to 25 points.
    rainfall_points = {
        "unknown": 0,
        "low": 3,
        "moderate": 10,
        "high": 18,
        "extreme": 25,
    }
    score += rainfall_points[rainfall]

    if rainfall == "unknown":
        data_gaps.append(
            "Rainfall intensity was not provided."
        )
    else:
        reasons.append(
            f"Rainfall indicator is {rainfall.replace('_', ' ')}."
        )

    # Population exposure contribution: up to 25 points.
    population_points = {
        "unknown": 0,
        "low": 3,
        "moderate": 10,
        "high": 18,
        "very_high": 25,
    }
    score += population_points[population]

    if population == "unknown":
        data_gaps.append(
            "Population exposure data was not provided."
        )
    else:
        reasons.append(
            f"Population exposure is {population.replace('_', ' ')}."
        )

    # Critical infrastructure contribution: 10 points.
    if infrastructure:
        score += 10
        reasons.append(
            "Critical infrastructure exposure was reported."
        )

    score = min(score, 100)

    # Assign an interpretable preliminary category.
    if score < 25:
        risk_level = "Low"
        priority = "Priority 4"
    elif score < 50:
        risk_level = "Moderate"
        priority = "Priority 3"
    elif score < 75:
        risk_level = "High"
        priority = "Priority 2"
    else:
        risk_level = "Very High"
        priority = "Priority 1"

    # Construct actions based on the preliminary risk category.
    recommended_actions = []

    if risk_level == "Low":
        recommended_actions = [
            {
                "priority": 1,
                "action": "Continue routine monitoring",
                "reason": "The heuristic score is currently low.",
            },
            {
                "priority": 2,
                "action": "Review updated rainfall and satellite data",
                "reason": "Conditions can change rapidly.",
            },
        ]

    elif risk_level == "Moderate":
        recommended_actions = [
            {
                "priority": 1,
                "action": "Verify the predicted flood extent",
                "reason": "Confirm the model output using local observations.",
            },
            {
                "priority": 2,
                "action": "Review local preparedness measures",
                "reason": "Prepare for possible worsening conditions.",
            },
            {
                "priority": 3,
                "action": "Monitor updated rainfall and satellite data",
                "reason": "Refresh the assessment as new data arrives.",
            },
        ]

    elif risk_level == "High":
        recommended_actions = [
            {
                "priority": 1,
                "action": "Request prompt field verification",
                "reason": "The heuristic score indicates elevated concern.",
            },
            {
                "priority": 2,
                "action": "Review response resources and access routes",
                "reason": "Identify potential resource and transport needs.",
            },
            {
                "priority": 3,
                "action": "Notify the responsible disaster-management team",
                "reason": "Request assessment by authorized personnel.",
            },
        ]

    else:
        recommended_actions = [
            {
                "priority": 1,
                "action": "Seek urgent verification by local authorities",
                "reason": "The heuristic score is in the highest category.",
            },
            {
                "priority": 2,
                "action": "Review emergency resources and safe access routes",
                "reason": "Support assessment and operational readiness.",
            },
            {
                "priority": 3,
                "action": "Check official alerts and ground conditions",
                "reason": "Use verified information for any protective action.",
            },
        ]

    if infrastructure:
        recommended_actions.insert(
            0,
            {
                "priority": 1,
                "action": "Verify the status of exposed critical infrastructure",
                "reason": "Reported infrastructure exposure needs local confirmation.",
            },
        )

        # Renumber actions after adding the infrastructure-specific item.
        for index, action in enumerate(recommended_actions, start=1):
            action["priority"] = index

    limitations = [
        "Flood pixel share is not a direct measurement of danger to people.",
        "The score is a prototype heuristic, not a calibrated probability of harm.",
        "The model has not been validated for operational emergency decisions.",
        "Do not use this response plan alone to issue warnings or order evacuations.",
    ]

    if data_gaps:
        limitations.append(
            "Missing exposure indicators may materially affect the score."
        )

    return {
        "project": "FloodSense",
        "status": "success",
        "assessment_type": "Experimental heuristic decision support",
        "risk_level": risk_level,
        "risk_score": score,
        "priority": priority,
        "predicted_flood_pixel_percentage": round(flood_share, 2),
        "input_indicators": {
            "rainfall_level": rainfall,
            "population_exposure": population,
            "critical_infrastructure_exposed": infrastructure,
        },
        "reasons": reasons,
        "data_gaps": data_gaps,
        "recommended_actions": recommended_actions,
        "limitations": limitations,
        "message": (
            "Preliminary risk category generated. Verify with reliable "
            "ground information and local authorities."
        ),
    }


@app.post("/response-plan")
def response_plan(request: ResponsePlanRequest):
    """Generate a preliminary response-prioritization assessment."""
    try:
        return calculate_response_plan(request)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate the response plan: {exc}",
        ) from exc
