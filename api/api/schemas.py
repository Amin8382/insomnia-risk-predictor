"""
api/api/schemas.py
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum

# Enums for categorical variables
class SexEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"

class EthnicityEnum(str, Enum):
    CAUCASIAN = "Caucasian"
    AFRICAN_AMERICAN = "African American"
    HISPANIC = "Hispanic"
    ASIAN = "Asian"
    OTHER = "Other"
    UNKNOWN = "Unknown"

class SmokingStatusEnum(str, Enum):
    CURRENT = "Current"
    PAST = "Past"
    NEVER = "Never"
    UNKNOWN = "Unknown"

# Request model for single prediction
class PatientData(BaseModel):
    """
    Patient data for insomnia risk prediction
    All fields are required - matching your dataset
    """
    # Demographics
    age: float = Field(
        ..., 
        ge=18, 
        le=120, 
        description="Patient age in years (18-120)",
        example=65.5
    )
    sex: SexEnum = Field(
        ..., 
        description="Patient sex",
        example="Female"
    )
    ethnicity: EthnicityEnum = Field(
        ..., 
        description="Patient ethnicity",
        example="Caucasian"
    )
    smoking_status: SmokingStatusEnum = Field(
        ..., 
        description="Smoking status",
        example="Never"
    )
    bmi: Optional[float] = Field(
        None, 
        ge=10, 
        le=80, 
        description="Body Mass Index",
        example=27.5
    )
    
    # Medical conditions (binary: 0 or 1)
    afib_or_flutter: int = Field(
        0, ge=0, le=1, 
        description="Atrial fibrillation or flutter (0=No, 1=Yes)",
        example=0
    )
    asthma: int = Field(
        0, ge=0, le=1, 
        description="Asthma (0=No, 1=Yes)",
        example=0
    )
    obesity: int = Field(
        0, ge=0, le=1, 
        description="Obesity (0=No, 1=Yes)",
        example=0
    )
    cancer: int = Field(
        0, ge=0, le=1, 
        description="Cancer (0=No, 1=Yes)",
        example=0
    )
    hypertension: int = Field(
        0, ge=0, le=1, 
        description="Hypertension (0=No, 1=Yes)",
        example=1
    )
    peripheral_vascular_disease: int = Field(
        0, ge=0, le=1, 
        description="Peripheral vascular disease (0=No, 1=Yes)",
        example=0
    )
    copd: int = Field(
        0, ge=0, le=1, 
        description="COPD (0=No, 1=Yes)",
        example=0
    )
    pneumonia: int = Field(
        0, ge=0, le=1, 
        description="Pneumonia (0=No, 1=Yes)",
        example=0
    )
    psychiatric_disorder: int = Field(
        0, ge=0, le=1, 
        description="Psychiatric disorder (0=No, 1=Yes)",
        example=0
    )
    lipid_metabolism_disorder: int = Field(
        0, ge=0, le=1, 
        description="Lipid metabolism disorder (0=No, 1=Yes)",
        example=0
    )
    cerebrovascular_disease: int = Field(
        0, ge=0, le=1, 
        description="Cerebrovascular disease (0=No, 1=Yes)",
        example=0
    )
    ckd_or_esrd: int = Field(
        0, ge=0, le=1, 
        description="CKD or ESRD (0=No, 1=Yes)",
        example=0
    )
    congestive_heart_failure: int = Field(
        0, ge=0, le=1, 
        description="Congestive heart failure (0=No, 1=Yes)",
        example=0
    )
    diabetes: int = Field(
        0, ge=0, le=1, 
        description="Diabetes (0=No, 1=Yes)",
        example=1
    )
    anxiety_or_depression: int = Field(
        0, ge=0, le=1, 
        description="Anxiety or depression (0=No, 1=Yes)",
        example=0
    )
    osteoporosis: int = Field(
        0, ge=0, le=1, 
        description="Osteoporosis (0=No, 1=Yes)",
        example=0
    )
    gastrointestinal_disorder: int = Field(
        0, ge=0, le=1, 
        description="Gastrointestinal disorder (0=No, 1=Yes)",
        example=0
    )
    renal_failure: int = Field(
        0, ge=0, le=1, 
        description="Renal failure (0=No, 1=Yes)",
        example=0
    )
    coronary_artery_disease: int = Field(
        0, ge=0, le=1, 
        description="Coronary artery disease (0=No, 1=Yes)",
        example=0
    )
    
    # Count features
    psych_note_count: int = Field(
        0, ge=0, 
        description="Number of psychiatry notes",
        example=1
    )
    insomnia_billing_code_count: int = Field(
        0, ge=0, 
        description="Number of insomnia billing codes",
        example=1
    )
    joint_disorder_billing_code_count: int = Field(
        0, ge=0, 
        description="Number of joint disorder billing codes",
        example=1
    )
    emr_fact_count: float = Field(
        0, ge=0, 
        description="EMR fact count",
        example=500.0
    )
    sleep_disorder_note_count: int = Field(
        0, ge=0, 
        description="Number of sleep disorder notes",
        example=2
    )
    anx_depr_billing_code_count: int = Field(
        0, ge=0, 
        description="Number of anxiety/depression billing codes",
        example=3
    )
    insomnia_rx_count: int = Field(
        0, ge=0, 
        description="Number of insomnia prescriptions",
        example=0
    )
    
    # Validators
    @field_validator('bmi')
    def validate_bmi(cls, v):
        if v is not None and (v < 10 or v > 80):
            raise ValueError('BMI must be between 10 and 80')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 65.5,
                "sex": "Female",
                "ethnicity": "Caucasian",
                "smoking_status": "Never",
                "bmi": 27.5,
                "hypertension": 1,
                "diabetes": 1,
                "sleep_disorder_note_count": 2,
                "insomnia_billing_code_count": 1,
                "anx_depr_billing_code_count": 3,
                "psych_note_count": 1,
                "insomnia_rx_count": 0,
                "joint_disorder_billing_code_count": 1,
                "emr_fact_count": 500.0,
                "afib_or_flutter": 0,
                "asthma": 0,
                "obesity": 0,
                "cancer": 0,
                "peripheral_vascular_disease": 0,
                "copd": 0,
                "pneumonia": 0,
                "psychiatric_disorder": 0,
                "lipid_metabolism_disorder": 0,
                "cerebrovascular_disease": 0,
                "ckd_or_esrd": 0,
                "congestive_heart_failure": 0,
                "osteoporosis": 0,
                "gastrointestinal_disorder": 0,
                "renal_failure": 0,
                "coronary_artery_disease": 0
            }
        }

# Request model for batch prediction
class BatchPatientData(BaseModel):
    """
    Batch of patients for multiple predictions
    """
    patients: List[PatientData] = Field(
        ..., 
        max_items=100, 
        description="List of patients (max 100)",
        example=[
            {
                "age": 65.5,
                "sex": "Female",
                "ethnicity": "Caucasian",
                "smoking_status": "Never",
                "bmi": 27.5,
                "hypertension": 1,
                "diabetes": 1,
                "sleep_disorder_note_count": 2,
                "insomnia_billing_code_count": 1,
                "anx_depr_billing_code_count": 3,
                "emr_fact_count": 500.0
            },
            {
                "age": 45.0,
                "sex": "Male",
                "ethnicity": "Hispanic",
                "smoking_status": "Current",
                "bmi": 32.0,
                "hypertension": 1,
                "diabetes": 0,
                "sleep_disorder_note_count": 5,
                "insomnia_billing_code_count": 3,
                "anx_depr_billing_code_count": 1,
                "emr_fact_count": 800.0
            }
        ]
    )

# Response models
class PredictionResponse(BaseModel):
    """
    Single prediction response
    """
    patient_id: Optional[str] = Field(
        None, 
        description="Optional patient identifier",
        example="pred_123456789"
    )
    insomnia_probability: float = Field(
        ..., 
        ge=0, le=1, 
        description="Probability of insomnia (0-1)",
        example=0.75
    )
    insomnia_class: int = Field(
        ..., 
        ge=0, le=1, 
        description="Predicted class (0: No insomnia, 1: Insomnia)",
        example=1
    )
    risk_level: str = Field(
        ..., 
        description="Risk level: Low/Medium/High",
        example="High"
    )
    
    @field_validator('risk_level')
    def validate_risk_level(cls, v):
        valid_levels = ['Low', 'Medium', 'High']
        if v not in valid_levels:
            raise ValueError(f'Risk level must be one of {valid_levels}')
        return v

class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response with summary statistics
    """
    predictions: List[PredictionResponse] = Field(
        ..., 
        description="List of individual predictions"
    )
    summary: Dict[str, Any] = Field(
        ..., 
        description="Summary statistics of batch predictions",
        example={
            "total_patients": 2,
            "successful": 2,
            "failed": 0,
            "mean_probability": 0.62,
            "std_probability": 0.18,
            "insomnia_rate": 0.5,
            "min_probability": 0.45,
            "max_probability": 0.78
        }
    )

class ModelInfoResponse(BaseModel):
    """
    Model metadata and performance information
    """
    model_name: str = Field(
        ..., 
        description="Name of the model",
        example="RandomForest"
    )
    model_version: str = Field(
        ..., 
        description="Model version",
        example="1.0.0"
    )
    features: List[str] = Field(
        ..., 
        description="List of feature names used by the model"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        None, 
        description="Feature importance scores (if available)"
    )
    performance_metrics: Dict[str, float] = Field(
        ..., 
        description="Model performance metrics",
        example={
            "f1_score": 0.70,
            "roc_auc": 0.80,
            "accuracy": 0.76
        }
    )
    training_date: str = Field(
        ..., 
        description="Date when model was trained",
        example="2026-02-18T20:58:37.627717"
    )

class HealthResponse(BaseModel):
    """
    Health check response
    """
    status: str = Field(
        ..., 
        description="API status",
        example="healthy"
    )
    model_loaded: bool = Field(
        ..., 
        description="Whether model is loaded and ready",
        example=True
    )
    api_version: str = Field(
        ..., 
        description="API version",
        example="1.0.0"
    )
    timestamp: str = Field(
        ..., 
        description="Current server time",
        example="2026-02-19T11:00:08.701014"
    )