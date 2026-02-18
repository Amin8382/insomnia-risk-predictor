"""
03_model_training.py
Model Training with MLflow Tracking for Insomnia Risk Predictor
Trains and tracks: Logistic Regression, Random Forest, Gradient Boosting
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import os
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)

class ModelTrainer:
    """
    Comprehensive model training with MLflow tracking
    """
    
    def __init__(self, data_path='data/processed/train_selected.csv'):
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports/figures', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
    def load_data(self):
        """
        Load the feature-selected dataset
        """
        logging.info(f"Loading data from {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        logging.info(f"Data shape: {df.shape}")
        
        # Separate features and target
        self.y = df['insomnia_class']
        self.X = df.drop('insomnia_class', axis=1)
        self.feature_names = list(self.X.columns)
        
        logging.info(f"Features: {len(self.feature_names)}")
        logging.info(f"Target distribution:\n{self.y.value_counts(normalize=True)}")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        """
        from sklearn.model_selection import train_test_split
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )
        
        logging.info(f"\nTrain set: {self.X_train.shape[0]} samples")
        logging.info(f"Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def setup_mlflow_experiment(self):
        """
        Setup MLflow experiment for tracking
        """
        experiment_name = "Insomnia_Risk_Prediction"
        
        # Try to get or create experiment
        try:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location="./mlruns"
            )
            logging.info(f"Created new experiment: {experiment_name}")
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
            logging.info(f"Using existing experiment: {experiment_name}")
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        return experiment_id
    
    def compute_metrics(self, y_true, y_pred, y_proba):
        """
        Compute comprehensive evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'pr_auc': average_precision_score(y_true, y_proba)
        }
        
        return metrics
    
    def cross_validate(self, model, model_name, cv_folds=5):
        """
        Perform cross-validation and return scores
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = {
            'accuracy': cross_val_score(model, self.X_train, self.y_train, 
                                       cv=cv, scoring='accuracy'),
            'f1': cross_val_score(model, self.X_train, self.y_train, 
                                 cv=cv, scoring='f1'),
            'roc_auc': cross_val_score(model, self.X_train, self.y_train, 
                                      cv=cv, scoring='roc_auc')
        }
        
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'cv_{metric}_mean'] = scores.mean()
            cv_results[f'cv_{metric}_std'] = scores.std()
            logging.info(f"CV {metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def plot_roc_curve(self, y_true, y_proba, model_name):
        """
        Plot and save ROC curve
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_true, y_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = f'reports/figures/roc_curve_{model_name}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_pr_curve(self, y_true, y_proba, model_name):
        """
        Plot and save Precision-Recall curve
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 
                label=f'PR curve (AUC = {average_precision_score(y_true, y_proba):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = f'reports/figures/pr_curve_{model_name}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Plot and save confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Insomnia', 'Insomnia'],
                   yticklabels=['No Insomnia', 'Insomnia'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = f'reports/figures/confusion_matrix_{model_name}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_feature_importance(self, importance_dict, model_name):
        """
        Plot and save feature importance
        """
        # Sort by importance
        sorted_items = sorted(importance_dict.items(), 
                             key=lambda x: abs(x[1]), reverse=True)
        features = [item[0] for item in sorted_items[:15]]
        importance = [item[1] for item in sorted_items[:15]]
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if x < 0 else 'green' for x in importance]
        plt.barh(range(len(importance)), importance, color=colors)
        plt.yticks(range(len(importance)), features)
        plt.xlabel('Importance')
        plt.title(f'Top 15 Feature Importances - {model_name}')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = f'reports/figures/feature_importance_{model_name}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def train_logistic_regression(self):
        """
        Train Logistic Regression with MLflow tracking
        """
        logging.info("\n" + "="*50)
        logging.info("TRAINING: Logistic Regression")
        logging.info("="*50)
        
        with mlflow.start_run(run_name="Logistic_Regression") as run:
            
            # Model parameters
            params = {
                'C': 1.0,
                'max_iter': 1000,
                'solver': 'lbfgs',
                'class_weight': 'balanced',
                'random_state': 42,
                'penalty': 'l2'
            }
            
            # Train model
            model = LogisticRegression(**params)
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Compute metrics
            metrics = self.compute_metrics(self.y_test, y_pred, y_proba)
            
            # Cross-validation
            cv_results = self.cross_validate(model, "Logistic_Regression")
            
            # Feature importance (coefficients)
            feature_importance = dict(zip(self.feature_names, 
                                         model.coef_[0]))
            
            # Log all to MLflow
            self._log_experiment(
                model=model,
                model_name="Logistic_Regression",
                params=params,
                metrics=metrics,
                cv_results=cv_results,
                y_true=self.y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                feature_importance=feature_importance
            )
            
            # Store model and results
            self.models['LogisticRegression'] = model
            self.results['LogisticRegression'] = {**metrics, **cv_results}
            
            logging.info(f"\n✅ Logistic Regression completed")
            logging.info(f"Test F1 Score: {metrics['f1_score']:.4f}")
            logging.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return model, metrics
    
    def train_random_forest(self):
        """
        Train Random Forest with MLflow tracking
        """
        logging.info("\n" + "="*50)
        logging.info("TRAINING: Random Forest")
        logging.info("="*50)
        
        with mlflow.start_run(run_name="Random_Forest") as run:
            
            # Model parameters
            params = {
                'n_estimators': 300,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Train model
            model = RandomForestClassifier(**params)
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Compute metrics
            metrics = self.compute_metrics(self.y_test, y_pred, y_proba)
            
            # Cross-validation
            cv_results = self.cross_validate(model, "Random_Forest")
            
            # Feature importance
            feature_importance = dict(zip(self.feature_names, 
                                         model.feature_importances_))
            
            # Log all to MLflow
            self._log_experiment(
                model=model,
                model_name="Random_Forest",
                params=params,
                metrics=metrics,
                cv_results=cv_results,
                y_true=self.y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                feature_importance=feature_importance
            )
            
            # Store model and results
            self.models['RandomForest'] = model
            self.results['RandomForest'] = {**metrics, **cv_results}
            
            logging.info(f"\n✅ Random Forest completed")
            logging.info(f"Test F1 Score: {metrics['f1_score']:.4f}")
            logging.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return model, metrics
    
    def train_gradient_boosting(self):
        """
        Train Gradient Boosting with MLflow tracking
        """
        logging.info("\n" + "="*50)
        logging.info("TRAINING: Gradient Boosting")
        logging.info("="*50)
        
        with mlflow.start_run(run_name="Gradient_Boosting") as run:
            
            # Model parameters
            params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'subsample': 0.8,
                'random_state': 42
            }
            
            # Train model
            model = GradientBoostingClassifier(**params)
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Compute metrics
            metrics = self.compute_metrics(self.y_test, y_pred, y_proba)
            
            # Cross-validation
            cv_results = self.cross_validate(model, "Gradient_Boosting")
            
            # Feature importance
            feature_importance = dict(zip(self.feature_names, 
                                         model.feature_importances_))
            
            # Log all to MLflow
            self._log_experiment(
                model=model,
                model_name="Gradient_Boosting",
                params=params,
                metrics=metrics,
                cv_results=cv_results,
                y_true=self.y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                feature_importance=feature_importance
            )
            
            # Store model and results
            self.models['GradientBoosting'] = model
            self.results['GradientBoosting'] = {**metrics, **cv_results}
            
            logging.info(f"\n✅ Gradient Boosting completed")
            logging.info(f"Test F1 Score: {metrics['f1_score']:.4f}")
            logging.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return model, metrics
    
    def _log_experiment(self, model, model_name, params, metrics, 
                        cv_results, y_true, y_pred, y_proba, 
                        feature_importance=None):
        """
        Internal method to log everything to MLflow
        """
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(cv_results)
        
        # Log dataset info
        mlflow.log_metric('train_samples', len(self.y_train))
        mlflow.log_metric('test_samples', len(self.y_test))
        mlflow.log_metric('n_features', len(self.feature_names))
        
        # Log tags
        mlflow.set_tag('model_type', model_name)
        mlflow.set_tag('problem_type', 'binary_classification')
        mlflow.set_tag('target', 'insomnia_class')
        
        # Generate and log plots
        roc_path = self.plot_roc_curve(y_true, y_proba, model_name)
        pr_path = self.plot_pr_curve(y_true, y_proba, model_name)
        cm_path = self.plot_confusion_matrix(y_true, y_pred, model_name)
        
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(pr_path)
        mlflow.log_artifact(cm_path)
        
        # Log feature importance if available
        if feature_importance:
            imp_path = self.plot_feature_importance(feature_importance, model_name)
            mlflow.log_artifact(imp_path)
            
            # Save importance as JSON
            imp_json_path = f'models/feature_importance_{model_name}.json'
            with open(imp_json_path, 'w') as f:
                json.dump(feature_importance, f, indent=2)
            mlflow.log_artifact(imp_json_path)
        
        # Log classification report
        report = classification_report(y_true, y_pred, 
                                      target_names=['No Insomnia', 'Insomnia'])
        report_path = f'reports/classification_report_{model_name}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        
        # Log model with signature
        signature = infer_signature(self.X_test[:5], model.predict(self.X_test[:5]))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            signature=signature,
            input_example=self.X_test[:5],
            registered_model_name=f"insomnia_{model_name}"
        )
        
        logging.info(f"✅ All artifacts logged to MLflow for {model_name}")
    
    def train_all_models(self):
        """
        Train all three models
        """
        logging.info("\n" + "="*60)
        logging.info("TRAINING ALL MODELS")
        logging.info("="*60)
        
        # Setup MLflow experiment
        self.setup_mlflow_experiment()
        
        # Train each model
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        
        logging.info("\n✅ All models trained successfully")
        
        return self.models
    
    def compare_models(self):
        """
        Compare all trained models and identify the best one
        """
        logging.info("\n" + "="*60)
        logging.info("MODEL COMPARISON")
        logging.info("="*60)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.results).T
        
        # Sort by F1 score (primary metric)
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        # Display comparison
        logging.info("\nModel Performance Comparison:")
        display_cols = ['accuracy', 'precision', 'recall', 'f1_score', 
                       'roc_auc', 'pr_auc', 'cv_f1_mean']
        
        for model in comparison_df.index:
            logging.info(f"\n{model}:")
            for col in display_cols:
                if col in comparison_df.columns:
                    value = comparison_df.loc[model, col]
                    logging.info(f"  {col}: {value:.4f}")
        
        # Identify best model
        self.best_model_name = comparison_df.index[0]
        self.best_model = self.models[self.best_model_name]
        best_f1 = comparison_df.loc[self.best_model_name, 'f1_score']
        
        logging.info(f"\n🏆 BEST MODEL: {self.best_model_name}")
        logging.info(f"   F1 Score: {best_f1:.4f}")
        logging.info(f"   ROC-AUC: {comparison_df.loc[self.best_model_name, 'roc_auc']:.4f}")
        
        # Save comparison
        comparison_df.to_csv('reports/model_comparison.csv')
        logging.info("\nComparison saved to reports/model_comparison.csv")
        
        # Plot comparison
        self.plot_model_comparison(comparison_df)
        
        return comparison_df, self.best_model_name
    
    def plot_model_comparison(self, comparison_df):
        """
        Plot model comparison bar chart
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.25
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (model_name, color) in enumerate(zip(comparison_df.index, colors)):
            values = [comparison_df.loc[model_name, m] for m in metrics]
            plt.bar(x + i*width, values, width, label=model_name, color=color, alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width, metrics, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, (model_name, color) in enumerate(zip(comparison_df.index, colors)):
            values = [comparison_df.loc[model_name, m] for m in metrics]
            for j, v in enumerate(values):
                plt.text(j + i*width, v + 0.01, f'{v:.3f}', 
                        ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('reports/figures/model_comparison.png', dpi=100)
        plt.close()
        
        logging.info("Model comparison plot saved to reports/figures/model_comparison.png")
    
    def save_best_model(self):
        """
        Save the best model for production
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run compare_models() first.")
        
        # Save model
        model_path = f'models/{self.best_model_name}_best.pkl'
        joblib.dump(self.best_model, model_path)
        
        # Also save as generic best_model.pkl
        best_path = 'models/best_model.pkl'
        joblib.dump(self.best_model, best_path)
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'f1_score': float(self.results[self.best_model_name]['f1_score']),
            'roc_auc': float(self.results[self.best_model_name]['roc_auc']),
            'features': self.feature_names,
            'n_features': len(self.feature_names),
            'training_date': datetime.now().isoformat()
        }
        
        with open('models/best_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"\n✅ Best model saved to {model_path}")
        logging.info(f"✅ Metadata saved to models/best_model_metadata.json")
        
        return model_path
    
    def run_training_pipeline(self):
        """
        Run the complete training pipeline
        """
        logging.info("="*60)
        logging.info("STARTING MODEL TRAINING PIPELINE")
        logging.info("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Split data
        self.split_data()
        
        # Step 3: Train all models
        self.train_all_models()
        
        # Step 4: Compare models
        self.compare_models()
        
        # Step 5: Save best model
        self.save_best_model()
        
        logging.info("="*60)
        logging.info("✅ MODEL TRAINING PIPELINE COMPLETED")
        logging.info("="*60)
        
        return self.best_model, self.best_model_name


# Execute training pipeline
if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer('data/processed/train_selected.csv')
    
    # Run full pipeline
    best_model, best_model_name = trainer.run_training_pipeline()
    
    print(f"\n🎉 Training complete! Best model: {best_model_name}")
    print("\nTo view MLflow UI, run: mlflow ui")
    print("Then open http://localhost:5000 in your browser")