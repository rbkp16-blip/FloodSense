# Cloud deployment

This Docker deployment serves the React dashboard and FastAPI on one URL. API routes are under /api. The original local development commands still work.

## Render
Create a Blueprint from this repository using render.yaml. Select the free plan only. Configure MODEL_URL as a direct HTTPS download URL for your trained flood_unet.pth, and MODEL_SHA256 as its SHA256 checksum. The model is downloaded at startup and verified before loading. Do not use dataset archives as the model URL. A missing model URL leaves inference unavailable.

The repository does not contain datasets or model weights. Upload the approximately 31 MB trained checkpoint separately to a model host or release before deploying. Free Render memory is limited to 512 MB; successful PyTorch inference must be verified on the deployed service, and may require a larger-memory host. Do not select a paid plan without authorization.

## Verification
Open /api/health and verify the trained model is loaded. Open the dashboard, upload a compatible seven-band GeoTIFF, and verify /api/predict returns a real model result. No sample datasets are bundled.

Free Render instances sleep after inactivity. Initial startup and model downloading can delay the first visit.
