"""
01_data_preprocessing.py
Data Preprocessing for Insomnia Risk Predictor
Handles: missing values, encoding, scaling for 1M patient records
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from datetime import datetime
import logging
import sys

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class DataPreprocessor:
    """
    Complete data preprocessing pipeline for insomnia prediction
    """
    
    def __init__(self, data_path='data/raw/patients_1m.csv'):
        self.data_path = data_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.binary_cols = None
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        logging.info("Directories checked/created successfully")
        
    def load_data(self, sample_size=None):
        """
        Load the 1M patient dataset
        Args:
            sample_size: if provided, load only sample_size rows (for testing)
        """
        logging.info(f"Loading data from {self.data_path}")
        
        # Check if file exists
        if not os.path.exists(self.data_path):
            logging.error(f"File not found: {self.data_path}")
            logging.info("Please ensure your data file is in data/raw/patients_1m.csv")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if sample_size:
            self.df = pd.read_csv(self.data_path, nrows=sample_size)
            logging.info(f"Loaded sample of {sample_size} rows")
        else:
            self.df = pd.read_csv(self.data_path)
            logging.info(f"Loaded full dataset: {len(self.df)} rows")
        
        logging.info(f"Dataset shape: {self.df.shape}")
        logging.info(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def identify_column_types(self):
        """
        Automatically identify column types based on your dataset structure
        """
        # Numerical columns (based on your dataset)
        self.numerical_cols = [
            'age', 'bmi', 'sleep_disorder_note_count',
            'insomnia_billing_code_count', 'anx_depr_billing_code_count',
            'psych_note_count', 'insomnia_rx_count',
            'joint_disorder_billing_code_count', 'emr_fact_count'
        ]
        
        # Categorical columns
        self.categorical_cols = ['sex', 'ethnicity', 'smoking_status']
        
        # Binary medical condition columns (0/1)
        self.binary_cols = [
            'afib_or_flutter', 'asthma', 'obesity', 'cancer',
            'hypertension', 'peripheral_vascular_disease', 'copd',
            'pneumonia', 'psychiatric_disorder', 'lipid_metabolism_disorder',
            'cerebrovascular_disease', 'ckd_or_esrd', 'congestive_heart_failure',
            'diabetes', 'anxiety_or_depression', 'osteoporosis',
            'gastrointestinal_disorder', 'renal_failure', 'coronary_artery_disease'
        ]
        
        # Target columns
        self.target_cols = ['insomnia_class', 'insomnia_probability']
        
        # Verify all columns exist
        existing_numerical = [col for col in self.numerical_cols if col in self.df.columns]
        existing_categorical = [col for col in self.categorical_cols if col in self.df.columns]
        existing_binary = [col for col in self.binary_cols if col in self.df.columns]
        
        logging.info(f"Identified {len(existing_numerical)} numerical columns")
        logging.info(f"Identified {len(existing_categorical)} categorical columns")
        logging.info(f"Identified {len(existing_binary)} binary medical columns")
        
        return existing_numerical, existing_categorical, existing_binary
    
    def analyze_missing_values(self):
        """
        Comprehensive missing value analysis
        """
        logging.info("\n=== MISSING VALUES ANALYSIS ===")
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        # Filter columns with missing values
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_df) > 0:
            logging.info(f"\nColumns with missing values:")
            for _, row in missing_df.iterrows():
                logging.info(f"  {row['Column']}: {row['Missing_Count']} "
                           f"({row['Missing_Percentage']:.2f}%)")
        else:
            logging.info("No missing values found!")
        
        return missing_df
    
    def handle_missing_values(self):
        """
        Handle missing values according to data type - FIXED VERSION
        No chained assignment, proper data type handling
        """
        logging.info("\n=== HANDLING MISSING VALUES ===")
        
        # 1. BMI - median imputation (since it might have outliers)
        if 'bmi' in self.df.columns:
            bmi_median = self.df['bmi'].median()
            self.df['bmi'] = self.df['bmi'].fillna(bmi_median)
            logging.info(f"BMI: imputed with median ({bmi_median:.2f})")
        
        # 2. Categorical columns - mode imputation
        for col in ['sex', 'ethnicity', 'smoking_status']:
            if col in self.df.columns:
                mode_val = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_val)
                logging.info(f"{col}: imputed with mode ({mode_val})")
        
        # 3. Count columns - 0 imputation (absence of records)
        count_cols = [
            'sleep_disorder_note_count', 'insomnia_billing_code_count',
            'anx_depr_billing_code_count', 'psych_note_count',
            'insomnia_rx_count', 'joint_disorder_billing_code_count'
        ]
        for col in count_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)
                logging.info(f"{col}: imputed with 0")
        
        # 4. EMR fact count - mean imputation
        if 'emr_fact_count' in self.df.columns:
            emr_mean = self.df['emr_fact_count'].mean()
            self.df['emr_fact_count'] = self.df['emr_fact_count'].fillna(emr_mean)
            logging.info(f"emr_fact_count: imputed with mean ({emr_mean:.2f})")
        
        # 5. insomnia_probability - median imputation (if exists)
        if 'insomnia_probability' in self.df.columns:
            prob_median = self.df['insomnia_probability'].median()
            self.df['insomnia_probability'] = self.df['insomnia_probability'].fillna(prob_median)
            logging.info(f"insomnia_probability: imputed with median ({prob_median:.2f})")
        
        # Verify no missing values remain
        remaining_missing = self.df.isnull().sum().sum()
        if remaining_missing == 0:
            logging.info("✅ All missing values handled successfully!")
        else:
            # Check which columns still have missing values
            remaining_cols = self.df.columns[self.df.isnull().any()].tolist()
            logging.warning(f"⚠️ {remaining_missing} missing values remain in: {remaining_cols}")
        
        return self.df
    
    def encode_categorical_variables(self):
        """
        Encode categorical variables using LabelEncoder
        """
        logging.info("\n=== ENCODING CATEGORICAL VARIABLES ===")
        
        for col in ['sex', 'ethnicity', 'smoking_status']:
            if col in self.df.columns:
                le = LabelEncoder()
                # Handle any potential NaN before encoding
                self.df[col] = self.df[col].fillna('Unknown')
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                logging.info(f"{col}: encoded into {len(le.classes_)} categories")
                logging.info(f"  Categories: {list(le.classes_)}")
        
        return self.df
    
    def prepare_features(self):
        """
        Prepare final feature set for modeling
        """
        logging.info("\n=== PREPARING FEATURE SET ===")
        
        # Get existing columns
        numerical = [col for col in self.numerical_cols if col in self.df.columns]
        binary = [col for col in self.binary_cols if col in self.df.columns]
        encoded = [f'{col}_encoded' for col in self.categorical_cols 
                  if f'{col}_encoded' in self.df.columns]
        
        # Combine all features
        self.feature_names = numerical + binary + encoded
        self.feature_names = [f for f in self.feature_names if f in self.df.columns]
        
        logging.info(f"Total features selected: {len(self.feature_names)}")
        logging.info("Features:")
        for i, f in enumerate(self.feature_names, 1):
            logging.info(f"  {i}. {f}")
        
        return self.feature_names
    
    def split_and_scale(self, test_size=0.2, random_state=42):
        """
        Split data and scale numerical features - FIXED VERSION
        Properly handles data types during scaling
        """
        logging.info("\n=== TRAIN-TEST SPLIT & SCALING ===")
        
        # Prepare feature matrix and target
        X = self.df[self.feature_names].copy()  # Make a copy to avoid warnings
        y = self.df['insomnia_class']
        
        # Log class distribution
        class_dist = y.value_counts(normalize=True)
        logging.info(f"Class distribution:\n  Class 0: {class_dist.get(0, 0):.3f}\n  Class 1: {class_dist.get(1, 0):.3f}")
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        logging.info(f"Train set: {X_train.shape[0]} samples")
        logging.info(f"Test set: {X_test.shape[0]} samples")
        
        # Identify numerical features that need scaling
        numerical_features = [col for col in self.feature_names 
                            if col in self.numerical_cols]
        
        if numerical_features:
            # Fit scaler on training data
            self.scaler.fit(X_train[numerical_features])
            
            # Transform both sets
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            # Apply scaling (this creates float columns)
            X_train_scaled[numerical_features] = self.scaler.transform(
                X_train[numerical_features]
            )
            X_test_scaled[numerical_features] = self.scaler.transform(
                X_test[numerical_features]
            )
            
            logging.info(f"Scaled {len(numerical_features)} numerical features")
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Ensure all features are float for consistency
        for col in numerical_features:
            X_train_scaled[col] = X_train_scaled[col].astype(float)
            X_test_scaled[col] = X_test_scaled[col].astype(float)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """
        Save all preprocessing objects for later use
        """
        logging.info(f"\nSaving preprocessor to {path}")
        
        preprocessor = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'binary_cols': self.binary_cols
        }
        
        joblib.dump(preprocessor, path)
        logging.info("✅ Preprocessor saved successfully")
        
        return path
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """
        Save processed datasets
        """
        logging.info("\n=== SAVING PROCESSED DATA ===")
        
        # Convert to DataFrames with proper column names
        train_df = pd.DataFrame(X_train, columns=self.feature_names)
        train_df['insomnia_class'] = y_train.values
        
        test_df = pd.DataFrame(X_test, columns=self.feature_names)
        test_df['insomnia_class'] = y_test.values
        
        # Save to CSV
        train_path = 'data/processed/train.csv'
        test_path = 'data/processed/test.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logging.info(f"Training data saved to {train_path}")
        logging.info(f"Test data saved to {test_path}")
        
        # Also save a sample for quick testing
        sample_path = 'data/processed/train_sample.csv'
        train_df.head(10000).to_csv(sample_path, index=False)
        logging.info(f"Sample (10k rows) saved to {sample_path}")
        
        return train_path, test_path
    
    def run_pipeline(self, sample_size=None):
        """
        Run the complete preprocessing pipeline
        """
        logging.info("="*60)
        logging.info("STARTING PREPROCESSING PIPELINE")
        logging.info("="*60)
        
        try:
            # Step 1: Load data
            self.load_data(sample_size)
            
            # Step 2: Identify column types
            self.identify_column_types()
            
            # Step 3: Analyze missing values
            self.analyze_missing_values()
            
            # Step 4: Handle missing values
            self.handle_missing_values()
            
            # Step 5: Encode categorical variables
            self.encode_categorical_variables()
            
            # Step 6: Prepare features
            self.prepare_features()
            
            # Step 7: Split and scale
            X_train, X_test, y_train, y_test = self.split_and_scale()
            
            # Step 8: Save preprocessor
            self.save_preprocessor()
            
            # Step 9: Save processed data
            self.save_processed_data(X_train, X_test, y_train, y_test)
            
            logging.info("="*60)
            logging.info("✅ PREPROCESSING PIPELINE COMPLETED")
            logging.info("="*60)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"❌ Error in preprocessing pipeline: {str(e)}")
            raise


# Execute preprocessing
if __name__ == "__main__":
    # First, check if data file exists
    data_path = 'data/raw/patients_1m.csv'
    
    if not os.path.exists(data_path):
        print(f"\n❌ Data file not found: {data_path}")
        print("\nPlease ensure:")
        print("1. Your data file is named 'patients_1m.csv'")
        print("2. It's placed in the 'data/raw/' folder")
        print("3. The path is correct")
        print("\nCurrent directory:", os.getcwd())
        print("\nFiles in data/raw/:")
        if os.path.exists('data/raw'):
            print(os.listdir('data/raw'))
        else:
            print("data/raw/ directory doesn't exist yet")
        
        response = input("\nDo you want to create the directory and continue? (y/n): ")
        if response.lower() == 'y':
            os.makedirs('data/raw', exist_ok=True)
            print("✅ Created data/raw/ directory")
            print("Please place your patients_1m.csv file there and run again")
        sys.exit(1)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_path)
    
    # Ask user if they want to run on full dataset or sample
    print("\n" + "="*60)
    print("PREPROCESSING OPTIONS")
    print("="*60)
    print("1. Run on full 1M dataset (may take 10-15 minutes)")
    print("2. Run on 100k sample for testing")
    print("3. Run on 10k sample for quick testing")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline(sample_size=None)
    elif choice == '2':
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline(sample_size=100000)
    elif choice == '3':
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline(sample_size=10000)
    else:
        print("Invalid choice. Running on 10k sample by default.")
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline(sample_size=10000)
    
    print("\n✅ Preprocessing complete!")
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print("\nNext step: Run feature selection")
    print("  python code/02_feature_selection.py")