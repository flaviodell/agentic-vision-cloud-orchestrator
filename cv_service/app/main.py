import logging
import io
import urllib.request

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.schemas import PredictionResponse, HealthResponse
from app.model import load_model, predict
from mangum import Mangum


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pet Breed CV Service",
    description="ResNet50 inference endpoint for Oxford Pets breed classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictURLRequest(BaseModel):
    image_url: str


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe."""
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_from_url(req: PredictURLRequest):
    """Accepts a JSON body with image_url, returns predicted breed + top-5."""
    try:
        with urllib.request.urlopen(req.image_url, timeout=10) as r:
            image_bytes = r.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot fetch image: {e}")

    try:
        result = predict(image_bytes)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    return result


@app.post("/predict-file", response_model=PredictionResponse)
async def predict_from_file(file: UploadFile = File(...)):
    """Accepts a multipart image upload, returns predicted breed + top-5."""
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported media type")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    try:
        result = predict(image_bytes)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    return result


# Lambda handler
handler = Mangum(app, lifespan="off", api_gateway_base_path="/prod")