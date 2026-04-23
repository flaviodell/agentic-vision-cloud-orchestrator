"""
CV tool — calls the Pet Breed CV Service (FastAPI/Lambda) from the agent.

The tool accepts an image URL, forwards it to the running cv_service endpoint,
and returns a JSON string with breed + confidence + top5.

Environment variable:
    CV_SERVICE_URL  (default: http://localhost:8000)
"""

import json
import logging
import os

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Timeout for the HTTP call to cv_service (seconds).
_HTTP_TIMEOUT = 30.0


def _get_cv_url() -> str:
    return os.getenv("CV_SERVICE_URL", "http://localhost:8000").rstrip("/")


@tool
def cv_predict(image_url: str) -> str:
    """
    Identify the breed of a cat or dog from an image URL.

    Use this tool whenever the user provides a URL pointing to a pet image
    and wants to know the breed. The tool returns the predicted breed,
    a confidence score (0-1), and the top-5 alternative predictions.

    Args:
        image_url: A publicly accessible URL of the pet image (JPEG or PNG).

    Returns:
        JSON string with fields: breed (str), confidence (float), top5 (list).
        On error, returns a JSON string with an "error" field.
    """
    base_url = _get_cv_url()
    endpoint = f"{base_url}/predict"

    logger.info(f"[cv_predict] Calling {endpoint} with url={image_url}")

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            response = client.post(endpoint, json={"image_url": image_url})
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError:
        msg = (
            f"Cannot connect to CV service at {base_url}. "
            "Check that CV_SERVICE_URL in .env points to the correct endpoint "
            "(AWS Lambda: https://8r6akcsyx5.execute-api.eu-south-1.amazonaws.com/prod "
            "or local Docker: http://localhost:8000)."
        )
        logger.error(f"[cv_predict] {msg}")
        return json.dumps({"error": msg})
    except httpx.HTTPStatusError as e:
        msg = f"CV service returned HTTP {e.response.status_code}: {e.response.text}"
        logger.error(f"[cv_predict] {msg}")
        return json.dumps({"error": msg})
    except Exception as e:
        logger.error(f"[cv_predict] Unexpected error: {e}")
        return json.dumps({"error": str(e)})

    logger.info(f"[cv_predict] Result: breed={data.get('breed')}, conf={data.get('confidence')}")
    return json.dumps(data)
