
from pathlib import Path
import json
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

EVALUATION_PATH = (
    PROJECT_ROOT / "docs" / "evaluation_report.json"
)

DEVICE = torch.device("cpu")

# Kuro Siwo test sample resolution
INPUT_SIZE = 224

FLOOD_THRESHOLD = 0.5

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


# ============================================================
# FASTAPI APPLICATION
# ============================================================

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


# ============================================================
# MODEL STATUS
# ============================================================

model = None

model_loaded = False

model_message = "Model has not been loaded."


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    global model, model_loaded, model_message

    try:

        model = UNet(
            in_channels=7,
            out_channels=1
        ).to(DEVICE)

        if not MODEL_PATH.exists():

            model = None
            model_loaded = False

            model_message = (
                "No trained checkpoint found at "
                "artifacts/flood_unet.pth."
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
                checkpoint.get(
                    "state_dict",
                    checkpoint
                ),
            )

        else:

            state_dict = checkpoint

        model.load_state_dict(state_dict)

        model.eval()

        model_loaded = True

        model_message = (
            "Trained U-Net checkpoint loaded successfully."
        )

        print("\n====================================")
        print("FLOODSENSE MODEL LOADED")
        print("====================================")
        print("Input channels: 7")
        print("Output channels: 1")
        print(f"Device: {DEVICE}")
        print(f"Checkpoint: {MODEL_PATH}")

    except Exception as exc:

        model = None
        model_loaded = False

        model_message = (
            f"Model loading failed: {exc}"
        )

        print(model_message)


load_model()


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "FloodSense",
        "status": "running",
        "version": app.version,
        "docs": "/docs",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

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


# ============================================================
# EVALUATION ENDPOINT
# ============================================================


@app.get("/evaluation")
def evaluation():

    paths = [
        ARTIFACTS_DIR / "evaluation_report.json",
        EVALUATION_PATH,
    ]

    report = None

    # Load the verified evaluation report.
    for path in paths:

        if path.is_file():

            try:
                with path.open(
                    "r",
                    encoding="utf-8"
                ) as handle:

                    data = json.load(handle)

                if isinstance(data, dict):
                    report = data
                    break

            except (OSError, ValueError):
                continue

    # Preserve the existing response when no report exists.
    if report is None:

        report = {
            "project": "FloodSense",
            "model": "U-Net",
            "dataset": "Kuro Siwo test_GRD",
            "evaluation_status": "report_not_available",
            "pixel_accuracy_percent": None,
            "flood_iou_percent": None,
            "precision_percent": None,
            "recall_percent": None,
            "f1_score_percent": None,
            "limitations": [
                "No verified evaluation report is available."
            ],
        }

    # Read training metadata from the saved checkpoint.
    try:

        if MODEL_PATH.is_file():

            checkpoint = torch.load(
                MODEL_PATH,
                map_location="cpu",
                weights_only=True
            )

            if isinstance(checkpoint, dict):

                report["training_samples"] = checkpoint.get(
                    "training_samples"
                )

                report["validation_samples"] = checkpoint.get(
                    "validation_samples"
                )

                report["epochs"] = checkpoint.get(
                    "epochs_completed"
                )

                report["input_channels"] = checkpoint.get(
                    "in_channels"
                )

                report["output_channels"] = checkpoint.get(
                    "out_channels"
                )

                report["best_validation_loss"] = checkpoint.get(
                    "best_validation_loss"
                )

                report["training_dataset"] = checkpoint.get(
                    "dataset"
                )

    except (OSError, RuntimeError, ValueError, TypeError) as exc:

        print(
            f"Warning: Could not load checkpoint metadata: {exc}"
        )

    return report



# ============================================================
# IMAGE PREPROCESSING
# ============================================================

# Exact input band order:
#
# 1. flood_vv
# 2. flood_vh
# 3. sec1_vv
# 4. sec1_vh
# 5. sec2_vv
# 6. sec2_vh
# 7. dem


def normalize_band(
    band: np.ndarray,
    valid_mask=None
) -> np.ndarray:

    """
    SAR normalization using finite pixels
    and the 2nd/98th percentiles.
    """

    band = band.astype(np.float32)

    valid = np.isfinite(band)

    if not valid.any():

        return np.zeros_like(
            band,
            dtype=np.float32
        )

    low, high = np.percentile(
        band[valid],
        [2, 98]
    )

    band = np.nan_to_num(
        band,
        nan=float(low),
        posinf=float(high),
        neginf=float(low),
    )

    if high > low:

        band = np.clip(
            (band - low) / (high - low),
            0.0,
            1.0
        )

    else:

        band[:] = 0.0

    return band.astype(np.float32)


def normalize_dem(
    dem: np.ndarray,
    valid_mask=None
) -> np.ndarray:

    """
    DEM normalization using mean and
    standard deviation.
    """

    dem = np.nan_to_num(
        dem,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    mean = float(dem.mean())

    std = float(dem.std())

    if std > 1e-6:

        dem = (
            dem - mean
        ) / std

    else:

        dem[:] = 0.0

    return dem.astype(np.float32)


# ============================================================
# IMAGE RESIZING
# ============================================================

def resize_to_model_input(
    image: np.ndarray
) -> torch.Tensor:

    """
    Convert a seven-channel image to
    a PyTorch tensor of shape:

    [1, 7, 224, 224]
    """

    tensor = torch.from_numpy(
        np.ascontiguousarray(image)
    ).float().unsqueeze(0)

    tensor = nn.functional.interpolate(
        tensor,
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return tensor


# ============================================================
# FLOOD PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    """
    Predict flood extent from a seven-band GeoTIFF.

    Expected channels:
        flood_vv
        flood_vh
        sec1_vv
        sec1_vh
        sec2_vv
        sec2_vh
        dem
    """

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not model_loaded or model is None:

        raise HTTPException(
            status_code=503,
            detail="Trained U-Net model is unavailable.",
        )

    # --------------------------------------------------------
    # CHECK FILE TYPE
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Please select a GeoTIFF file.",
        )

    if Path(file.filename).suffix.lower() not in {
        ".tif",
        ".tiff"
    }:

        raise HTTPException(
            status_code=400,
            detail="Only .tif and .tiff files are supported.",
        )

    # --------------------------------------------------------
    # READ UPLOADED FILE
    # --------------------------------------------------------

    contents = await file.read(
        MAX_UPLOAD_BYTES + 1
    )

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

    # --------------------------------------------------------
    # READ SEVEN-BAND GEOTIFF
    # --------------------------------------------------------

    try:

        with rasterio.MemoryFile(contents) as memory_file:

            with memory_file.open() as dataset:

                if dataset.count != 7:

                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Seven-band GeoTIFF required. "
                            "Expected six SAR bands and one DEM band."
                        ),
                    )

                width = dataset.width

                height = dataset.height

                image = dataset.read(
                    indexes=list(range(1, 8)),
                    out_dtype="float32",
                )

                band_masks = dataset.read_masks(
                    indexes=list(range(1, 8))
                )

                valid_mask = np.all(
                    band_masks > 0,
                    axis=0
                )

                valid_mask &= np.all(
                    np.isfinite(image),
                    axis=0
                )

                if not np.any(valid_mask):

                    raise HTTPException(
                        status_code=400,
                        detail="No valid pixels found.",
                    )

                # --------------------------------------------
                # NORMALIZE SIX SAR CHANNELS
                # --------------------------------------------

                normalized_bands = []

                for index in range(6):

                    normalized_bands.append(
                        normalize_band(
                            image[index]
                        )
                    )

                # --------------------------------------------
                # NORMALIZE DEM CHANNEL
                # --------------------------------------------

                normalized_bands.append(
                    normalize_dem(
                        image[6]
                    )
                )

                # --------------------------------------------
                # COMBINE SEVEN CHANNELS
                # --------------------------------------------

                normalized_image = np.stack(
                    normalized_bands,
                    axis=0
                ).astype(np.float32)

                valid_pixel_count = int(
                    valid_mask.sum()
                )

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read GeoTIFF: {exc}",
        ) from exc

    # --------------------------------------------------------
    # PREPARE MODEL INPUT
    # --------------------------------------------------------

    input_tensor = resize_to_model_input(
        normalized_image
    ).to(DEVICE)

    valid_tensor = torch.from_numpy(
        valid_mask.astype(np.float32)
    )[None, None]

    resized_valid = nn.functional.interpolate(
        valid_tensor,
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="nearest",
    )[0, 0].numpy() > 0.5

    # --------------------------------------------------------
    # RUN U-NET PREDICTION
    # --------------------------------------------------------

    try:

        model.eval()

        with torch.inference_mode():

            logits = model(
                input_tensor
            )

            probabilities = torch.sigmoid(
                logits
            )[0, 0].cpu().numpy()

        probabilities = np.nan_to_num(
            probabilities,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        probabilities = np.clip(
            probabilities,
            0.0,
            1.0
        )

        probabilities[~resized_valid] = 0.0

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {exc}",
        ) from exc

    # --------------------------------------------------------
    # CALCULATE FLOOD EXTENT
    # --------------------------------------------------------

    flood_mask = (
        probabilities >= FLOOD_THRESHOLD
    ) & resized_valid

    predicted_flood_pixel_count = int(
        flood_mask.sum()
    )

    inference_valid_pixel_count = int(
        resized_valid.sum()
    )

    if inference_valid_pixel_count > 0:

        flood_percentage = (
            predicted_flood_pixel_count
            / inference_valid_pixel_count
        ) * 100.0

    else:

        flood_percentage = 0.0

    # --------------------------------------------------------
    # RETURN PREDICTION
    # --------------------------------------------------------

    return {
        "project": "FloodSense",
        "status": "success",
        "filename": file.filename,
        "width": width,
        "height": height,
        "input_bands": 7,
        "inference_width": INPUT_SIZE,
        "inference_height": INPUT_SIZE,
        "threshold": FLOOD_THRESHOLD,
        "valid_pixel_count": valid_pixel_count,
        "inference_valid_pixel_count": (
            inference_valid_pixel_count
        ),
        "predicted_flood_pixel_count": (
            predicted_flood_pixel_count
        ),
        "predicted_flood_pixel_percentage": round(
            flood_percentage,
            2
        ),
        "probability_map": (
            probabilities.round(4).tolist()
        ),
        "model": "U-Net",
        "model_checkpoint_loaded": True,
        "warning": (
            "Research prototype only. "
            "Not validated for operational flood warnings."
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
        description=(
            "Predicted flood share expressed as a percentage."
        ),
    )

    rainfall_level: RainfallLevel = "unknown"

    population_exposure: PopulationExposure = "unknown"

    critical_infrastructure_exposed: bool = False


# ============================================================
# RESPONSE PRIORITIZATION LOGIC
# ============================================================

def calculate_response_plan(
    request: ResponsePlanRequest
):

    """
    Experimental heuristic risk assessment.

    This is not an official emergency
    classification or calibrated risk probability.
    """

    flood_share = (
        request.predicted_flood_pixel_percentage
    )

    rainfall = request.rainfall_level

    population = request.population_exposure

    infrastructure = (
        request.critical_infrastructure_exposed
    )

    score = 0

    reasons = []

    data_gaps = []

    # --------------------------------------------------------
    # FLOOD EXTENT SCORE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RAINFALL SCORE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # POPULATION EXPOSURE SCORE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CRITICAL INFRASTRUCTURE SCORE
    # --------------------------------------------------------

    if infrastructure:

        score += 10

        reasons.append(
            "Critical infrastructure exposure was reported."
        )

    score = min(score, 100)

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RECOMMENDED ACTIONS
    # --------------------------------------------------------

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
                "reason": (
                    "Confirm the model output using local observations."
                ),
            },
            {
                "priority": 2,
                "action": "Review local preparedness measures",
                "reason": (
                    "Prepare for possible worsening conditions."
                ),
            },
            {
                "priority": 3,
                "action": "Monitor updated rainfall and satellite data",
                "reason": (
                    "Refresh the assessment as new data arrives."
                ),
            },
        ]

    elif risk_level == "High":

        recommended_actions = [
            {
                "priority": 1,
                "action": "Request prompt field verification",
                "reason": (
                    "The heuristic score indicates elevated concern."
                ),
            },
            {
                "priority": 2,
                "action": "Review response resources and access routes",
                "reason": (
                    "Identify potential resource and transport needs."
                ),
            },
            {
                "priority": 3,
                "action": (
                    "Notify the responsible disaster-management team"
                ),
                "reason": (
                    "Request assessment by authorized personnel."
                ),
            },
        ]

    else:

        recommended_actions = [
            {
                "priority": 1,
                "action": (
                    "Seek urgent verification by local authorities"
                ),
                "reason": (
                    "The heuristic score is in the highest category."
                ),
            },
            {
                "priority": 2,
                "action": (
                    "Review emergency resources and safe access routes"
                ),
                "reason": (
                    "Support assessment and operational readiness."
                ),
            },
            {
                "priority": 3,
                "action": (
                    "Check official alerts and ground conditions"
                ),
                "reason": (
                    "Use verified information for any protective action."
                ),
            },
        ]

    # --------------------------------------------------------
    # INFRASTRUCTURE PRIORITY
    # --------------------------------------------------------

    if infrastructure:

        recommended_actions.insert(
            0,
            {
                "priority": 1,
                "action": (
                    "Verify the status of exposed critical infrastructure"
                ),
                "reason": (
                    "Reported infrastructure exposure needs "
                    "local confirmation."
                ),
            },
        )

        for index, action in enumerate(
            recommended_actions,
            start=1
        ):

            action["priority"] = index

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    limitations = [
        (
            "Flood pixel share is not a direct measurement "
            "of danger to people."
        ),
        (
            "The score is a prototype heuristic, not a "
            "calibrated probability of harm."
        ),
        (
            "The model has not been validated for "
            "operational emergency decisions."
        ),
        (
            "Do not use this response plan alone to issue "
            "warnings or order evacuations."
        ),
    ]

    if data_gaps:

        limitations.append(
            "Missing exposure indicators may materially "
            "affect the score."
        )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "project": "FloodSense",
        "status": "success",
        "assessment_type": (
            "Experimental heuristic decision support"
        ),
        "risk_level": risk_level,
        "risk_score": score,
        "priority": priority,
        "predicted_flood_pixel_percentage": round(
            flood_share,
            2
        ),
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
            "Preliminary risk category generated. "
            "Verify with reliable ground information "
            "and local authorities."
        ),
    }


# ============================================================
# RESPONSE PLAN ENDPOINT
# ============================================================

@app.post("/response-plan")
def response_plan(
    request: ResponsePlanRequest
):

    try:

        return calculate_response_plan(
            request
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not generate the response plan: {exc}"
            ),
        ) from exc


# ============================================================
# END OF FLOODSENSE BACKEND
# ============================================================
