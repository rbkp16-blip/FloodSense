# FloodSense

An AI-Driven Multimodal Framework for Flood Early Warning, Extent Mapping, Risk Assessment and Intelligent Response.

Developed by Sagnik Halder as an M.Tech CSE project.

## Components
- React/Vite dashboard in frontend/
- FastAPI backend in backend/main.py
- PyTorch U-Net with seven input channels for SAR and elevation data
- Training, dataset preparation and evaluation scripts in backend/
- Evaluation report in docs/

## Local setup
Use Python 3.11 and Node.js. Create a virtual environment and install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn python-multipart numpy rasterio torch
uvicorn backend.main:app --reload
```

In a separate terminal:

```powershell
cd frontend
npm ci
npm run dev
```

The frontend defaults to http://127.0.0.1:8000 for the API. Set VITE_API_URL in a local frontend/.env if needed.

## Excluded files
Datasets, test inputs, trained model weights, virtual environments, node_modules and local .env files are intentionally excluded. Supply your own compatible data under data/ and trained checkpoint at artifacts/flood_unet.pth to run model inference. This repository contains source code and does not include the trained model.

Risk and response outputs are research prototype decision support.
