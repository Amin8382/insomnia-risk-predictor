"""
05_hyperparameter_tuning.py
Hyperparameter Optimization for Insomnia Risk Predictor
GridSearchCV and RandomizedSearchCV for all three models
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, make_scorer
import joblib
import os
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hyperparameter_tuning.log'),
        logging.StreamHandler()
    ]
)

class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning for all models
    """
    
    def __init__(self, data_path='data/processed/train_selected.csv'):
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.best_models = {}
        self.search_results = {}
        
        # Create directories
        os.makedirs('models/tuned', exist_ok=True)
        os.makedirs('reports/tuning', exist_ok=True)
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
        Setup MLflow experiment for tuning
        """
        experiment_name = "Insomnia_Hyperparameter_Tuning"
        
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
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
    
    def plot_tuning_results(self, cv_results, model_name, param_name, metric='mean_test_score'):
        """
        Plot tuning results for a specific parameter
        """
        plt.figure(figsize=(12, 6))
        
        # Extract data
        params = cv_results['params']
        scores = cv_results[metric]
        
        # If parameter is numerical, create line plot
        if all(isinstance(p[param_name], (int, float)) for p in params if param_name in p):
            param_values = [p.get(param_name, None) for p in params]
            param_values = [v for v in param_values if v is not None]
            unique_values = sorted(set(param_values))
            
            mean_scores = []
            std_scores = []
            
            for val in unique_values:
                indices = [i for i, p in enumerate(params) if p.get(param_name) == val]
                if indices:
                    mean_scores.append(np.mean([scores[i] for i in indices]))
                    std_scores.append(np.std([scores[i] for i in indices]))
            
            plt.errorbar(unique_values, mean_scores, yerr=std_scores, 
                        marker='o', capsize=5, capthick=2)
            plt.xlabel(param_name)
            
        else:
            # Categorical parameter - bar plot
            param_values = [str(p.get(param_name, 'N/A')) for p in params]
            unique_values = list(set(param_values))
            
            mean_scores = []
            for val in unique_values:
                indices = [i for i, pv in enumerate(param_values) if pv == str(val)]
                mean_scores.append(np.mean([scores[i] for i in indices]))
            
            bars = plt.bar(range(len(unique_values)), mean_scores)
            plt.xticks(range(len(unique_values)), unique_values, rotation=45, ha='right')
            plt.xlabel(param_name)
        
        plt.ylabel('F1 Score (CV)')
        plt.title(f'{model_name} - Tuning Results for {param_name}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f'reports/tuning/{model_name}_{param_name}_tuning.png'
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        return plot_path
    
    def plot_learning_curve(self, model, model_name):
        """
        Plot learning curve for the best model
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X_train, self.y_train,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='f1'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        
        plt.fill_between(train_sizes, train_mean - train_std, 
                        train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, test_mean - test_std, 
                        test_mean + test_std, alpha=0.1, color='orange')
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
        
        plt.xlabel('Training examples')
        plt.ylabel('F1 Score')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f'reports/tuning/{model_name}_learning_curve.png'
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        return plot_path
    
    def tune_logistic_regression(self, n_iter=50):
        """
        Hyperparameter tuning for Logistic Regression
        """
        logging.info("\n" + "="*60)
        logging.info("TUNING: Logistic Regression")
        logging.info("="*60)
        
        with mlflow.start_run(run_name="Logistic_Regression_Tuning") as run:
            
            # Define parameter grid
            param_distributions = {
                'C': np.logspace(-4, 4, 20),
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [100, 200, 500, 1000],
                'class_weight': [None, 'balanced'],
                'fit_intercept': [True, False]
            }
            
            # Filter valid combinations
            valid_params = []
            for solver in param_distributions['solver']:
                for penalty in param_distributions['penalty']:
                    # Check compatibility
                    if solver in ['newton-cg', 'lbfgs', 'sag'] and penalty in ['l2', None]:
                        valid_params.append((solver, penalty))
                    elif solver == 'liblinear' and penalty in ['l1', 'l2']:
                        valid_params.append((solver, penalty))
                    elif solver == 'saga' and penalty in ['l1', 'l2', 'elasticnet', None]:
                        valid_params.append((solver, penalty))
            
            # Create base model
            base_model = LogisticRegression(random_state=42)
            
            # Setup cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Randomized search
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions={
                    'C': param_distributions['C'],
                    'penalty': [p for s, p in valid_params],
                    'solver': [s for s, p in valid_params],
                    'max_iter': param_distributions['max_iter'],
                    'class_weight': param_distributions['class_weight'],
                    'fit_intercept': param_distributions['fit_intercept']
                },
                n_iter=n_iter,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                random_state=42,
                verbose=1,
                return_train_score=True
            )
            
            # Fit random search
            logging.info("Starting randomized search...")
            random_search.fit(self.X_train, self.y_train)
            
            # Log parameters
            mlflow.log_params({
                'tuning_method': 'RandomizedSearchCV',
                'n_iterations': n_iter,
                'cv_folds': 5
            })
            
            # Log best parameters
            best_params = random_search.best_params_
            mlflow.log_params({f'best_{k}': v for k, v in best_params.items()})
            
            # Log best score
            mlflow.log_metric('best_cv_f1', random_search.best_score_)
            
            # Evaluate on test set
            best_model = random_search.best_estimator_
            y_pred = best_model.predict(self.X_test)
            y_proba = best_model.predict_proba(self.X_test)[:, 1]
            
            from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
            test_f1 = f1_score(self.y_test, y_pred)
            test_roc_auc = roc_auc_score(self.y_test, y_proba)
            test_accuracy = accuracy_score(self.y_test, y_pred)
            
            mlflow.log_metrics({
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc,
                'test_accuracy': test_accuracy
            })
            
            logging.info(f"\nBest parameters: {best_params}")
            logging.info(f"Best CV F1: {random_search.best_score_:.4f}")
            logging.info(f"Test F1: {test_f1:.4f}")
            logging.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
            
            # Plot tuning results for main parameters
            for param in ['C', 'penalty', 'solver']:
                if param in best_params:
                    plot_path = self.plot_tuning_results(
                        random_search.cv_results_, 
                        'LogisticRegression', 
                        param
                    )
                    mlflow.log_artifact(plot_path)
            
            # Plot learning curve
            lc_path = self.plot_learning_curve(best_model, 'LogisticRegression')
            mlflow.log_artifact(lc_path)
            
            # Save results
            self.best_models['LogisticRegression'] = best_model
            self.search_results['LogisticRegression'] = {
                'best_params': best_params,
                'best_score': random_search.best_score_,
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc,
                'cv_results': str(random_search.cv_results_)
            }
            
            # Log model
            signature = infer_signature(self.X_test[:5], best_model.predict(self.X_test[:5]))
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="tuned_logistic_regression",
                signature=signature,
                input_example=self.X_test[:5]
            )
            
            # Save model locally
            joblib.dump(best_model, 'models/tuned/logistic_regression_tuned.pkl')
            
            logging.info("✅ Logistic Regression tuning completed")
            
            return best_model, random_search.best_params_, random_search.best_score_
    
    def tune_random_forest(self, n_iter=50):
        """
        Hyperparameter tuning for Random Forest
        """
        logging.info("\n" + "="*60)
        logging.info("TUNING: Random Forest")
        logging.info("="*60)
        
        with mlflow.start_run(run_name="Random_Forest_Tuning") as run:
            
            # Define parameter grid
            param_distributions = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [10, 20, 30, 40, 50, None],
                'min_samples_split': [2, 5, 10, 15, 20],
                'min_samples_leaf': [1, 2, 4, 8, 12],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            }
            
            # Create base model
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
            # Setup cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Randomized search
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                random_state=42,
                verbose=1,
                return_train_score=True
            )
            
            # Fit random search
            logging.info("Starting randomized search...")
            random_search.fit(self.X_train, self.y_train)
            
            # Log parameters
            mlflow.log_params({
                'tuning_method': 'RandomizedSearchCV',
                'n_iterations': n_iter,
                'cv_folds': 5
            })
            
            # Log best parameters
            best_params = random_search.best_params_
            mlflow.log_params({f'best_{k}': v for k, v in best_params.items()})
            
            # Log best score
            mlflow.log_metric('best_cv_f1', random_search.best_score_)
            
            # Evaluate on test set
            best_model = random_search.best_estimator_
            y_pred = best_model.predict(self.X_test)
            y_proba = best_model.predict_proba(self.X_test)[:, 1]
            
            from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
            test_f1 = f1_score(self.y_test, y_pred)
            test_roc_auc = roc_auc_score(self.y_test, y_proba)
            test_accuracy = accuracy_score(self.y_test, y_pred)
            
            mlflow.log_metrics({
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc,
                'test_accuracy': test_accuracy
            })
            
            logging.info(f"\nBest parameters: {best_params}")
            logging.info(f"Best CV F1: {random_search.best_score_:.4f}")
            logging.info(f"Test F1: {test_f1:.4f}")
            logging.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
            
            # Plot tuning results for main parameters
            for param in ['n_estimators', 'max_depth', 'min_samples_split']:
                if param in best_params:
                    plot_path = self.plot_tuning_results(
                        random_search.cv_results_, 
                        'RandomForest', 
                        param
                    )
                    mlflow.log_artifact(plot_path)
            
            # Plot learning curve
            lc_path = self.plot_learning_curve(best_model, 'RandomForest')
            mlflow.log_artifact(lc_path)
            
            # Feature importance
            feature_importance = dict(zip(self.feature_names, 
                                         best_model.feature_importances_))
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sorted_imp = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:15]
            features, importances = zip(*sorted_imp)
            
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance')
            plt.title('Random Forest - Top 15 Feature Importances (Tuned)')
            plt.tight_layout()
            
            imp_path = 'reports/tuning/rf_tuned_importance.png'
            plt.savefig(imp_path, dpi=100)
            plt.close()
            mlflow.log_artifact(imp_path)
            
            # Save results
            self.best_models['RandomForest'] = best_model
            self.search_results['RandomForest'] = {
                'best_params': best_params,
                'best_score': random_search.best_score_,
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc,
                'feature_importance': feature_importance,
                'cv_results': str(random_search.cv_results_)
            }
            
            # Log model
            signature = infer_signature(self.X_test[:5], best_model.predict(self.X_test[:5]))
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="tuned_random_forest",
                signature=signature,
                input_example=self.X_test[:5]
            )
            
            # Save model locally
            joblib.dump(best_model, 'models/tuned/random_forest_tuned.pkl')
            
            logging.info("✅ Random Forest tuning completed")
            
            return best_model, random_search.best_params_, random_search.best_score_
    
    def tune_gradient_boosting(self, n_iter=50):
        """
        Hyperparameter tuning for Gradient Boosting
        """
        logging.info("\n" + "="*60)
        logging.info("TUNING: Gradient Boosting")
        logging.info("="*60)
        
        with mlflow.start_run(run_name="Gradient_Boosting_Tuning") as run:
            
            # Define parameter grid
            param_distributions = {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 7, 9, 11],
                'min_samples_split': [2, 5, 10, 15, 20],
                'min_samples_leaf': [1, 2, 4, 8, 12],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Create base model
            base_model = GradientBoostingClassifier(random_state=42)
            
            # Setup cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Randomized search
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                random_state=42,
                verbose=1,
                return_train_score=True
            )
            
            # Fit random search
            logging.info("Starting randomized search...")
            random_search.fit(self.X_train, self.y_train)
            
            # Log parameters
            mlflow.log_params({
                'tuning_method': 'RandomizedSearchCV',
                'n_iterations': n_iter,
                'cv_folds': 5
            })
            
            # Log best parameters
            best_params = random_search.best_params_
            mlflow.log_params({f'best_{k}': v for k, v in best_params.items()})
            
            # Log best score
            mlflow.log_metric('best_cv_f1', random_search.best_score_)
            
            # Evaluate on test set
            best_model = random_search.best_estimator_
            y_pred = best_model.predict(self.X_test)
            y_proba = best_model.predict_proba(self.X_test)[:, 1]
            
            from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
            test_f1 = f1_score(self.y_test, y_pred)
            test_roc_auc = roc_auc_score(self.y_test, y_proba)
            test_accuracy = accuracy_score(self.y_test, y_pred)
            
            mlflow.log_metrics({
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc,
                'test_accuracy': test_accuracy
            })
            
            logging.info(f"\nBest parameters: {best_params}")
            logging.info(f"Best CV F1: {random_search.best_score_:.4f}")
            logging.info(f"Test F1: {test_f1:.4f}")
            logging.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
            
            # Plot tuning results for main parameters
            for param in ['n_estimators', 'learning_rate', 'max_depth']:
                if param in best_params:
                    plot_path = self.plot_tuning_results(
                        random_search.cv_results_, 
                        'GradientBoosting', 
                        param
                    )
                    mlflow.log_artifact(plot_path)
            
            # Plot learning curve
            lc_path = self.plot_learning_curve(best_model, 'GradientBoosting')
            mlflow.log_artifact(lc_path)
            
            # Feature importance
            feature_importance = dict(zip(self.feature_names, 
                                         best_model.feature_importances_))
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sorted_imp = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:15]
            features, importances = zip(*sorted_imp)
            
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance')
            plt.title('Gradient Boosting - Top 15 Feature Importances (Tuned)')
            plt.tight_layout()
            
            imp_path = 'reports/tuning/gb_tuned_importance.png'
            plt.savefig(imp_path, dpi=100)
            plt.close()
            mlflow.log_artifact(imp_path)
            
            # Save results
            self.best_models['GradientBoosting'] = best_model
            self.search_results['GradientBoosting'] = {
                'best_params': best_params,
                'best_score': random_search.best_score_,
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc,
                'feature_importance': feature_importance,
                'cv_results': str(random_search.cv_results_)
            }
            
            # Log model
            signature = infer_signature(self.X_test[:5], best_model.predict(self.X_test[:5]))
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="tuned_gradient_boosting",
                signature=signature,
                input_example=self.X_test[:5]
            )
            
            # Save model locally
            joblib.dump(best_model, 'models/tuned/gradient_boosting_tuned.pkl')
            
            logging.info("✅ Gradient Boosting tuning completed")
            
            return best_model, random_search.best_params_, random_search.best_score_
    
    def compare_tuned_models(self):
        """
        Compare all tuned models
        """
        logging.info("\n" + "="*60)
        logging.info("COMPARING TUNED MODELS")
        logging.info("="*60)
        
        comparison = []
        for model_name, results in self.search_results.items():
            comparison.append({
                'Model': model_name,
                'Best CV F1': results['best_score'],
                'Test F1': results['test_f1'],
                'Test ROC-AUC': results['test_roc_auc'],
                'Improvement': results['test_f1'] - results['best_score']
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('Test F1', ascending=False)
        
        logging.info("\nTuned Models Comparison:")
        for _, row in comparison_df.iterrows():
            logging.info(f"\n{row['Model']}:")
            logging.info(f"  CV F1: {row['Best CV F1']:.4f}")
            logging.info(f"  Test F1: {row['Test F1']:.4f}")
            logging.info(f"  Test ROC-AUC: {row['Test ROC-AUC']:.4f}")
            logging.info(f"  Improvement: {row['Improvement']:.4f}")
        
        # Identify best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = self.best_models[best_model_name]
        
        logging.info(f"\n🏆 BEST TUNED MODEL: {best_model_name}")
        logging.info(f"   Test F1: {comparison_df.iloc[0]['Test F1']:.4f}")
        
        # Save comparison
        comparison_df.to_csv('reports/tuning/tuned_models_comparison.csv', index=False)
        
        # Save best model as final model
        joblib.dump(best_model, 'models/best_tuned_model.pkl')
        
        # Save metadata
        metadata = {
            'best_model': best_model_name,
            'test_f1': float(comparison_df.iloc[0]['Test F1']),
            'test_roc_auc': float(comparison_df.iloc[0]['Test ROC-AUC']),
            'all_results': self.search_results,
            'tuning_date': datetime.now().isoformat()
        }
        
        with open('models/best_tuned_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info("\n✅ Best tuned model saved to models/best_tuned_model.pkl")
        
        # Plot comparison
        self.plot_tuned_comparison(comparison_df)
        
        return comparison_df, best_model_name, best_model
    
    def plot_tuned_comparison(self, comparison_df):
        """
        Plot comparison of tuned models
        """
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.35
        
        plt.bar(x - width/2, comparison_df['Best CV F1'], width, 
                label='CV F1', alpha=0.8, color='skyblue')
        plt.bar(x + width/2, comparison_df['Test F1'], width,
                label='Test F1', alpha=0.8, color='lightcoral')
        
        plt.xlabel('Model')
        plt.ylabel('F1 Score')
        plt.title('Tuned Models - CV vs Test F1 Score')
        plt.xticks(x, comparison_df['Model'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, row in comparison_df.iterrows():
            plt.text(i - width/2, row['Best CV F1'] + 0.01, 
                    f"{row['Best CV F1']:.3f}", ha='center', fontsize=9)
            plt.text(i + width/2, row['Test F1'] + 0.01,
                    f"{row['Test F1']:.3f}", ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('reports/tuning/tuned_models_comparison.png', dpi=100)
        plt.close()
        
        logging.info("✅ Tuned models comparison plot saved")
    
    def run_tuning_pipeline(self, n_iter=30):
        """
        Run complete hyperparameter tuning pipeline
        """
        logging.info("="*60)
        logging.info("STARTING HYPERPARAMETER TUNING PIPELINE")
        logging.info("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Split data
        self.split_data()
        
        # Step 3: Setup MLflow
        self.setup_mlflow_experiment()
        
        # Step 4: Tune each model
        logging.info("\n" + "="*60)
        logging.info("TUNING ALL MODELS")
        logging.info("="*60)
        
        self.tune_logistic_regression(n_iter=n_iter)
        self.tune_random_forest(n_iter=n_iter)
        self.tune_gradient_boosting(n_iter=n_iter)
        
        # Step 5: Compare tuned models
        comparison_df, best_model_name, best_model = self.compare_tuned_models()
        
        logging.info("="*60)
        logging.info("✅ HYPERPARAMETER TUNING PIPELINE COMPLETED")
        logging.info("="*60)
        
        return best_model, best_model_name, comparison_df


def main():
    """
    Main function to run hyperparameter tuning
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning')
    parser.add_argument('--iterations', type=int, default=30,
                       help='Number of iterations for random search (default: 30)')
    parser.add_argument('--model', type=str, choices=['lr', 'rf', 'gb', 'all'],
                       default='all', help='Model to tune (default: all)')
    
    args = parser.parse_args()
    
    # Initialize tuner
    tuner = HyperparameterTuner('data/processed/train_selected.csv')
    
    # Load and split data
    tuner.load_data()
    tuner.split_data()
    
    # Setup MLflow
    tuner.setup_mlflow_experiment()
    
    # Tune specified model(s)
    if args.model == 'all' or args.model == 'lr':
        tuner.tune_logistic_regression(n_iter=args.iterations)
    
    if args.model == 'all' or args.model == 'rf':
        tuner.tune_random_forest(n_iter=args.iterations)
    
    if args.model == 'all' or args.model == 'gb':
        tuner.tune_gradient_boosting(n_iter=args.iterations)
    
    # Compare if all models were tuned
    if args.model == 'all':
        tuner.compare_tuned_models()
    
    print("\n✅ Tuning complete!")
    print("\nTo view results in MLflow UI:")
    print("  mlflow ui")
    print("  Then open http://localhost:5000")
    print("\nTuned models saved in:")
    print("  models/tuned/")
    print("\nReports saved in:")
    print("  reports/tuning/")


if __name__ == "__main__":
    main()