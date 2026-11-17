"""Serve the existing API and React dashboard from one cloud service."""
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
checkpoint = ROOT / "artifacts" / "flood_unet.pth"
model_url = os.environ.get("MODEL_URL", "https://github.com/rbkp16-blip/FloodSense/releases/download/model-v1/flood_unet.pth")
if model_url and not checkpoint.exists():
    if not model_url.startswith("https://"):
        raise RuntimeError("MODEL_URL must use HTTPS")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_suffix(".download")
    digest = hashlib.sha256()
    try:
        with urlopen(model_url, timeout=120) as response, partial.open("wb") as target:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 100 * 1024 * 1024:
                    raise RuntimeError("Model download exceeds 100 MB")
                digest.update(chunk)
                target.write(chunk)
        expected = os.environ.get("MODEL_SHA256", "338b25b1f77342d2b7a929dbb5589ae6faba790cff6998fceeb0fb1d67e01357")
        if not expected or digest.hexdigest() != expected.lower():
            raise RuntimeError("Model SHA256 missing or does not match")
        partial.replace(checkpoint)
    finally:
        partial.unlink(missing_ok=True)

import torch
torch.set_num_threads(1)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.main import app as api

app = FastAPI(title="FloodSense")
app.mount("/api", api)
app.mount("/", StaticFiles(directory=ROOT / "frontend" / "dist", html=True), name="dashboard")
