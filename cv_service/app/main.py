import logging
import time
import urllib.request

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.schemas import PredictionResponse, HealthResponse
from app.model import load_model, predict
from app.metrics import (
    INFERENCE_LATENCY,
    PREDICTIONS_TOTAL,
    CONFIDENCE_SCORE,
    MODEL_LOAD_TIME,
    metrics_app,
)
from mangum import Mangum


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the ResNet50 model at startup and record elapsed time."""
    logger.info("Pre-loading ResNet50 model...")
    t0 = time.time()
    load_model()
    elapsed = time.time() - t0
    MODEL_LOAD_TIME.set(elapsed)
    logger.info(f"Model loaded in {elapsed:.2f}s")
    yield
    # Shutdown logic would go here if needed


app = FastAPI(
    title="Pet Breed CV Service",
    description="ResNet50 inference endpoint for Oxford Pets breed classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics at /metrics
app.mount("/metrics", metrics_app)


class PredictURLRequest(BaseModel):
    image_url: str


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe."""
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_from_url(req: PredictURLRequest):
    """Accepts a JSON body with image_url, returns predicted breed + top-5."""
    t0 = time.time()

    try:
        with urllib.request.urlopen(req.image_url, timeout=10) as r:
            image_bytes = r.read()
    except Exception as e:
        PREDICTIONS_TOTAL.labels(breed="unknown", status="error").inc()
        raise HTTPException(status_code=400, detail=f"Cannot fetch image: {e}")

    try:
        result = predict(image_bytes)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        PREDICTIONS_TOTAL.labels(breed="unknown", status="error").inc()
        raise HTTPException(status_code=500, detail="Inference failed")

    elapsed = time.time() - t0
    breed = result.get("breed", "unknown")
    confidence = result.get("confidence", 0.0)

    INFERENCE_LATENCY.observe(elapsed)
    PREDICTIONS_TOTAL.labels(breed=breed, status="success").inc()
    CONFIDENCE_SCORE.observe(confidence)

    logger.info(f"Predicted: {breed} ({confidence:.3f}) in {elapsed:.3f}s")
    return result


@app.post("/predict-file", response_model=PredictionResponse)
async def predict_from_file(file: UploadFile = File(...)):
    """Accepts a multipart image upload, returns predicted breed + top-5."""
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported media type")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    t0 = time.time()
    try:
        result = predict(image_bytes)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        PREDICTIONS_TOTAL.labels(breed="unknown", status="error").inc()
        raise HTTPException(status_code=500, detail="Inference failed")

    elapsed = time.time() - t0
    breed = result.get("breed", "unknown")
    confidence = result.get("confidence", 0.0)

    INFERENCE_LATENCY.observe(elapsed)
    PREDICTIONS_TOTAL.labels(breed=breed, status="success").inc()
    CONFIDENCE_SCORE.observe(confidence)

    return result


# Lambda handler
handler = Mangum(app, lifespan="off", api_gateway_base_path="/prod")
