"""
api/api/routers/predict.py
Prediction endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional
import time
import numpy as np

from api.api.schemas import (
    PatientData, BatchPatientData, PredictionResponse,
    BatchPredictionResponse, ModelInfoResponse
)
from api.api.dependencies import (
    get_prediction_service, get_model_loader,
    PredictionService, ModelLoader
)

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/", response_model=PredictionResponse)
async def predict_single(
    patient: PatientData,
    background_tasks: BackgroundTasks,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict insomnia risk for a single patient
    
    - **patient**: Patient data including demographics and medical history
    - Returns: Probability and risk level
    """
    start_time = time.time()
    
    try:
        # Make prediction
        result = service.predict(patient.dict())
        
        # Add patient_id if not present
        if 'patient_id' not in result:
            result['patient_id'] = f"pred_{int(time.time())}"
        
        # Log prediction in background
        background_tasks.add_task(
            log_prediction,
            patient.dict(),
            result,
            time.time() - start_time
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch: BatchPatientData,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict insomnia risk for multiple patients (max 100)
    
    - **patients**: List of patient data
    - Returns: List of predictions with summary statistics
    """
    try:
        # Convert to dict list
        patients_data = [p.dict() for p in batch.patients]
        
        # Make batch predictions
        predictions = service.predict_batch(patients_data)
        
        # Calculate summary statistics
        valid_preds = [p for p in predictions if 'error' not in p]
        if valid_preds:
            probabilities = [p['insomnia_probability'] for p in valid_preds]
            classes = [p['insomnia_class'] for p in valid_preds]
            
            summary = {
                'total_patients': len(predictions),
                'successful': len(valid_preds),
                'failed': len(predictions) - len(valid_preds),
                'mean_probability': float(np.mean(probabilities)),
                'std_probability': float(np.std(probabilities)),
                'insomnia_rate': float(np.mean(classes)),
                'min_probability': float(np.min(probabilities)),
                'max_probability': float(np.max(probabilities))
            }
        else:
            summary = {
                'total_patients': len(predictions),
                'successful': 0,
                'failed': len(predictions),
                'mean_probability': 0,
                'std_probability': 0,
                'insomnia_rate': 0,
                'min_probability': 0,
                'max_probability': 0
            }
        
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**p) for p in predictions if 'error' not in p],
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(
    model_loader: ModelLoader = Depends(get_model_loader)
):
    """
    Get information about the loaded model
    
    Returns: Model name, features, performance metrics, etc.
    """
    if not model_loader.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    metadata = model_loader.metadata or {}
    
    return ModelInfoResponse(
        model_name=metadata.get('model_name', 'Unknown'),
        model_version=metadata.get('version', '1.0.0'),
        features=model_loader.feature_names or [],
        feature_importance=metadata.get('feature_importance', {}),
        performance_metrics=metadata.get('performance_metrics', {}),
        training_date=metadata.get('training_date', 'Unknown')
    )

# Background task
async def log_prediction(input_data: dict, output_data: dict, latency: float):
    """
    Log prediction for monitoring (can be extended to save to database)
    """
    # In a real application, you might save this to a database
    print(f"Prediction logged - Latency: {latency:.3f}s")
    print(f"Input: {input_data.get('age', 'N/A')}")
    print(f"Output: {output_data.get('insomnia_class', 'N/A')}")