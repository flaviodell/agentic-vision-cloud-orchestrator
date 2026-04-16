from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    breed: str
    confidence: float
    top5: List[dict]  # [{"breed": str, "confidence": float}]

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
