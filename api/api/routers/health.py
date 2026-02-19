"""
api/api/routers/health.py
Health check endpoints
"""

from fastapi import APIRouter, Depends
from datetime import datetime
from api.api.dependencies import get_model_loader, ModelLoader
from api.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check(
    model_loader: ModelLoader = Depends(get_model_loader)
):
    """
    Health check endpoint
    Returns API status and model information
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_ready,
        api_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@router.get("/ready")
async def readiness_check(
    model_loader: ModelLoader = Depends(get_model_loader)
):
    """
    Readiness probe for Kubernetes/Docker
    """
    if model_loader.is_ready:
        return {"status": "ready"}
    return {"status": "not ready"}

@router.get("/live")
async def liveness_check():
    """
    Liveness probe for Kubernetes/Docker
    """
    return {"status": "alive"}