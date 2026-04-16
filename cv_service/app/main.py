from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.schemas import PredictionResponse, HealthResponse
from app.model import load_model, predict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model at startup (warm-up)
    logger.info("Loading CV model...")
    load_model()
    logger.info("CV model ready.")
    yield


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


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe."""
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict_breed(file: UploadFile = File(...)):
    """
    Accepts a JPEG/PNG image, returns the predicted breed + top-5.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported media type")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    try:
        result = predict(image_bytes)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    return result
