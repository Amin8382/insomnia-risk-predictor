"""
api/api/app.py
Main FastAPI application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import uvicorn
from pathlib import Path

# Import routers
from api.api.routers import predict, health
from api.api.dependencies import ModelLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events handler
    - Startup: Load model
    - Shutdown: Cleanup
    """
    # Startup
    logger.info("🚀 Starting Insomnia Risk Predictor API")
    
    # Pre-load model
    model_loader = ModelLoader()
    if model_loader.is_ready:
        logger.info("✅ Model loaded successfully")
    else:
        logger.warning("⚠️ Model not loaded - predictions will fail")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Insomnia Risk Predictor API")

# Create FastAPI app with enhanced Swagger config
app = FastAPI(
    title="😴 Insomnia Risk Predictor API",
    description="""
    ## 🎯 Purpose
    Predict insomnia risk based on patient demographics and medical history.
    
    ## 🔬 Model Information
    - **Algorithm**: Random Forest Classifier
    - **Features**: 31 clinical and demographic features
    - **Performance**: F1 Score ~0.70, ROC-AUC ~0.80
    
    ## 📊 Available Endpoints
    * `GET /health` - Check API status
    * `GET /predict/model-info` - Get model information
    * `POST /predict/` - Single patient prediction
    * `POST /predict/batch` - Batch predictions (up to 100 patients)
    
    ## 🚀 Quick Start
    1. Use the **`/predict/`** endpoint with a single patient
    2. Check **`/predict/model-info`** to see model details
    3. Try batch predictions with multiple patients
    
    ## 📝 Notes
    - All medical condition fields default to 0 if not specified
    - BMI is optional (will use default if missing)
    - Risk levels: Low (<0.3), Medium (0.3-0.6), High (>0.6)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    contact={
        "name": "Insomnia Risk Predictor Team",
        "email": "team@insomnia-predictor.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "Health",
            "description": "Health check endpoints for monitoring"
        },
        {
            "name": "Prediction",
            "description": "Insomnia risk prediction endpoints"
        }
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(predict.router)

# Root endpoint
@app.get("/", 
         summary="API Root",
         description="Returns basic API information",
         tags=["Root"])
async def root():
    """
    Root endpoint - API information
    """
    return {
        "name": "Insomnia Risk Predictor API",
        "version": "1.0.0",
        "description": "Predict insomnia risk from patient data",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "health": "/health",
            "model_info": "/predict/model-info",
            "single_prediction": "/predict/",
            "batch_prediction": "/predict/batch"
        }
    }

# Exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Run the app
if __name__ == "__main__":
    uvicorn.run(
        "api.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )