"""
api/api/dependencies.py
Dependency injection for FastAPI
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional
import json
import os

# IMPORTANT: Add these FastAPI imports
from fastapi import Depends

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Singleton class to load and manage the ML model
    """
    _instance = None
    _model = None
    _preprocessor = None
    _metadata = None
    _feature_names = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        """Load model, preprocessor, and metadata"""
        try:
            # Load preprocessor
            preprocessor_path = Path('models/preprocessor.pkl')
            if preprocessor_path.exists():
                self._preprocessor = joblib.load(preprocessor_path)
                logger.info("✅ Preprocessor loaded successfully")
            else:
                logger.warning(f"⚠️ Preprocessor not found at {preprocessor_path}")
                self._preprocessor = None
            
            # Load best model
            model_path = Path('models/best_model.pkl')
            if model_path.exists():
                self._model = joblib.load(model_path)
                logger.info("✅ Model loaded successfully")
            else:
                logger.warning(f"⚠️ Model not found at {model_path}")
                self._model = None
            
            # Load metadata
            metadata_path = Path('models/best_model_metadata.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self._metadata = json.load(f)
                logger.info("✅ Metadata loaded successfully")
            else:
                logger.warning(f"⚠️ Metadata not found at {metadata_path}")
                self._metadata = None
            
            # Get feature names from model
            if self._model is not None:
                self._feature_names = list(self._model.feature_names_in_)
                logger.info(f"📊 Model expects {len(self._feature_names)} features")
            elif self._preprocessor and 'feature_names' in self._preprocessor:
                self._feature_names = self._preprocessor['feature_names']
            elif self._metadata and 'features' in self._metadata:
                self._feature_names = self._metadata['features']
            else:
                self._feature_names = None
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            self._model = None
            self._preprocessor = None
    
    @property
    def model(self):
        return self._model
    
    @property
    def preprocessor(self):
        return self._preprocessor
    
    @property
    def metadata(self):
        return self._metadata
    
    @property
    def feature_names(self):
        return self._feature_names
    
    @property
    def is_ready(self):
        return self._model is not None

class PredictionService:
    """
    Service class for making predictions
    """
    
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.logger = logging.getLogger(__name__)
        # Mapping for categorical encodings (based on your training)
        self.sex_mapping = {'Female': 0, 'Male': 1}
        self.ethnicity_mapping = {
            'African American': 0,
            'Asian': 1,
            'Caucasian': 2,
            'Hispanic': 3,
            'Other': 4,
            'Unknown': 5
        }
        self.smoking_mapping = {
            'Current': 0,
            'Never': 1,
            'Past': 2,
            'Unknown': 3
        }
    
    def prepare_features(self, patient_data: dict) -> pd.DataFrame:
        """
        Prepare features for prediction (must match training)
        """
        self.logger.info("="*50)
        self.logger.info("Preparing features for prediction")
        
        # Get expected feature names from model
        expected_features = self.model_loader.feature_names
        self.logger.info(f"Model expects {len(expected_features)} features")
        
        # Create a dictionary with ALL expected features initialized to 0
        feature_dict = {feature: 0 for feature in expected_features}
        
        # 1. Copy numerical and binary features directly
        numerical_binary_features = [
            'age', 'bmi', 'sleep_disorder_note_count', 'insomnia_billing_code_count',
            'anx_depr_billing_code_count', 'psych_note_count', 'insomnia_rx_count',
            'joint_disorder_billing_code_count', 'emr_fact_count', 'afib_or_flutter',
            'asthma', 'obesity', 'cancer', 'hypertension', 'peripheral_vascular_disease',
            'copd', 'pneumonia', 'psychiatric_disorder', 'lipid_metabolism_disorder',
            'cerebrovascular_disease', 'ckd_or_esrd', 'congestive_heart_failure',
            'diabetes', 'anxiety_or_depression', 'osteoporosis', 'gastrointestinal_disorder',
            'renal_failure', 'coronary_artery_disease'
        ]
        
        for feature in numerical_binary_features:
            if feature in patient_data and patient_data[feature] is not None:
                feature_dict[feature] = patient_data[feature]
                self.logger.debug(f"Set {feature} = {patient_data[feature]}")
        
        # 2. Handle BMI (if present)
        if 'bmi' in feature_dict and feature_dict['bmi'] == 0:
            feature_dict['bmi'] = 30.0  # Default median
            self.logger.info("BMI defaulted to 30.0")
        
        # 3. Encode categorical variables
        # Sex encoding
        if 'sex' in patient_data and patient_data['sex'] is not None:
            sex_val = patient_data['sex']
            if isinstance(sex_val, str):
                feature_dict['sex_encoded'] = self.sex_mapping.get(sex_val, 0)
                self.logger.info(f"Encoded sex: {sex_val} -> {feature_dict['sex_encoded']}")
        
        # Ethnicity encoding
        if 'ethnicity' in patient_data and patient_data['ethnicity'] is not None:
            ethnicity_val = patient_data['ethnicity']
            if isinstance(ethnicity_val, str):
                feature_dict['ethnicity_encoded'] = self.ethnicity_mapping.get(ethnicity_val, 5)
                self.logger.info(f"Encoded ethnicity: {ethnicity_val} -> {feature_dict['ethnicity_encoded']}")
        
        # Smoking status encoding
        if 'smoking_status' in patient_data and patient_data['smoking_status'] is not None:
            smoking_val = patient_data['smoking_status']
            if isinstance(smoking_val, str):
                feature_dict['smoking_status_encoded'] = self.smoking_mapping.get(smoking_val, 3)
                self.logger.info(f"Encoded smoking_status: {smoking_val} -> {feature_dict['smoking_status_encoded']}")
        
        # Create DataFrame with features in the EXACT order expected
        final_df = pd.DataFrame([feature_dict])[expected_features]
        
        self.logger.info(f"Final feature vector shape: {final_df.shape}")
        self.logger.info(f"Features prepared: {list(final_df.columns)}")
        
        # Log first few values for debugging
        sample_values = final_df.values[0][:10]
        self.logger.info(f"First 10 values: {sample_values}")
        
        return final_df
    
    def predict(self, patient_data: dict) -> dict:
        """
        Make prediction for a single patient
        """
        self.logger.info("="*50)
        self.logger.info("Making prediction for single patient")
        
        if not self.model_loader.is_ready:
            raise ValueError("Model not loaded")
        
        try:
            # Prepare features
            X = self.prepare_features(patient_data)
            
            # Make prediction
            probability = float(self.model_loader.model.predict_proba(X)[0, 1])
            prediction = int(self.model_loader.model.predict(X)[0])
            
            self.logger.info(f"Prediction probability: {probability:.4f}")
            self.logger.info(f"Prediction class: {prediction}")
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.6:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            result = {
                'insomnia_probability': probability,
                'insomnia_class': prediction,
                'risk_level': risk_level
            }
            
            self.logger.info(f"Risk level: {risk_level}")
            self.logger.info("="*50)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, patients_data: list) -> list:
        """
        Make predictions for multiple patients
        """
        self.logger.info(f"Making batch predictions for {len(patients_data)} patients")
        
        results = []
        for i, patient in enumerate(patients_data):
            try:
                self.logger.info(f"Processing patient {i+1}")
                pred = self.predict(patient)
                pred['patient_id'] = f"batch_{i+1}"
                results.append(pred)
            except Exception as e:
                self.logger.error(f"Error predicting for patient {i+1}: {str(e)}")
                results.append({
                    'patient_id': f"batch_{i+1}",
                    'error': str(e)
                })
        
        successful = len([r for r in results if 'error' not in r])
        self.logger.info(f"Batch complete: {successful}/{len(results)} successful")
        
        return results

# Dependency to get model loader
def get_model_loader() -> ModelLoader:
    return ModelLoader()

# Dependency to get prediction service
def get_prediction_service(
    model_loader: ModelLoader = Depends(get_model_loader)
) -> PredictionService:
    return PredictionService(model_loader)