"""
Prometheus metrics for the CV inference service.

Exposes:
  - cv_inference_latency_seconds  (Histogram)  — per-request latency
  - cv_predictions_total          (Counter)    — requests by breed + status
  - cv_confidence_score           (Histogram)  — distribution of top-1 confidence
  - cv_model_load_seconds         (Gauge)      — time taken to load model at startup
"""

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# Latency histogram: buckets tuned for CPU inference (50ms–10s)
INFERENCE_LATENCY = Histogram(
    "cv_inference_latency_seconds",
    "End-to-end latency of /predict endpoint (image download + model inference)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Counter: total predictions, labelled by breed and outcome
PREDICTIONS_TOTAL = Counter(
    "cv_predictions_total",
    "Total prediction requests processed",
    ["breed", "status"],  # status: success | error
)

# Histogram: confidence score distribution
CONFIDENCE_SCORE = Histogram(
    "cv_confidence_score",
    "Top-1 confidence score returned by the model",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)

# Gauge: model cold-start load time (set once at startup)
MODEL_LOAD_TIME = Gauge(
    "cv_model_load_seconds",
    "Seconds taken to download and load the ResNet50 checkpoint at startup",
)

# ASGI sub-app that serves GET /metrics
metrics_app = make_asgi_app()
