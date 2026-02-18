"""
04_mlflow_tracking.py
MLflow Tracking Configuration and Management
Setup, UI instructions, experiment analysis, and model registry
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
import subprocess
import webbrowser
import time
from datetime import datetime
import socket
import sys

# Setup logging with UTF-8 encoding for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mlflow_tracking.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class MLflowManager:
    """
    Manage MLflow tracking, experiments, and model registry
    """
    
    def __init__(self, tracking_uri="sqlite:///mlflow.db"):
        self.tracking_uri = tracking_uri
        self.client = None
        self.experiment_name = "Insomnia_Risk_Prediction"
        self.experiment_id = None
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        
        # Create directories
        os.makedirs('mlruns', exist_ok=True)
        os.makedirs('reports/mlflow', exist_ok=True)
        
    def setup_experiment(self):
        """
        Setup or get existing experiment
        """
        logging.info("\n=== SETTING UP MLFLOW EXPERIMENT ===")
        
        try:
            # Try to create new experiment
            self.experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location="./mlruns"
            )
            logging.info(f"✅ Created new experiment: {self.experiment_name}")
            logging.info(f"Experiment ID: {self.experiment_id}")
            
        except Exception as e:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
                logging.info(f"✅ Using existing experiment: {self.experiment_name}")
                logging.info(f"Experiment ID: {self.experiment_id}")
                logging.info(f"Artifact Location: {experiment.artifact_location}")
                logging.info(f"Lifecycle Stage: {experiment.lifecycle_stage}")
            else:
                logging.error(f"❌ Error setting up experiment: {str(e)}")
                raise
        
        # Set as active experiment
        mlflow.set_experiment(self.experiment_name)
        
        return self.experiment_id
    
    def check_mlflow_server(self, port=5000):
        """
        Check if MLflow server is running
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            logging.info(f"✅ MLflow server is running on port {port}")
            return True
        else:
            logging.warning(f"⚠️ MLflow server is NOT running on port {port}")
            return False
    
    def start_mlflow_ui(self, port=5000, backend_store_uri="sqlite:///mlflow.db"):
        """
        Start MLflow UI server
        """
        logging.info("\n=== STARTING MLFLOW UI ===")
        
        if self.check_mlflow_server(port):
            logging.info(f"MLflow UI already running at http://localhost:{port}")
            webbrowser.open(f'http://localhost:{port}')
            return
        
        try:
            # Start MLflow UI in background
            command = [
                'mlflow', 'ui',
                '--port', str(port),
                '--backend-store-uri', backend_store_uri
            ]
            
            logging.info(f"Starting MLflow UI with command: {' '.join(command)}")
            
            # On Windows, use start command to run in background
            if os.name == 'nt':  # Windows
                subprocess.Popen(command, shell=True, 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            else:  # Linux/Mac
                subprocess.Popen(command, stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL, 
                               start_new_session=True)
            
            # Wait for server to start
            time.sleep(3)
            
            if self.check_mlflow_server(port):
                logging.info(f"✅ MLflow UI started at http://localhost:{port}")
                webbrowser.open(f'http://localhost:{port}')
            else:
                logging.error("❌ Failed to start MLflow UI")
                
        except Exception as e:
            logging.error(f"❌ Error starting MLflow UI: {str(e)}")
            logging.info("\nTry running manually:")
            logging.info(f"  mlflow ui --port {port} --backend-store-uri {backend_store_uri}")
    
    def list_experiments(self):
        """
        List all experiments
        """
        logging.info("\n=== MLFLOW EXPERIMENTS ===")
        
        experiments = self.client.search_experiments()
        
        for exp in experiments:
            logging.info(f"\nExperiment: {exp.name}")
            logging.info(f"  ID: {exp.experiment_id}")
            logging.info(f"  Artifact Location: {exp.artifact_location}")
            logging.info(f"  Lifecycle Stage: {exp.lifecycle_stage}")
            logging.info(f"  Tags: {exp.tags}")
            
            # Get runs for this experiment
            runs = self.client.search_runs(exp.experiment_id)
            logging.info(f"  Runs: {len(runs)}")
        
        return experiments
    
    def get_experiment_runs(self):
        """
        Get all runs for the current experiment
        """
        logging.info(f"\n=== RUNS FOR EXPERIMENT: {self.experiment_name} ===")
        
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["attributes.start_time DESC"]
        )
        
        run_data = []
        for run in runs:
            run_info = {
                'run_id': run.info.run_id,
                'run_name': run.info.run_name,
                'status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time/1000),
                'duration': (run.info.end_time - run.info.start_time)/1000 if run.info.end_time else None,
                'model_type': run.data.tags.get('model_type', 'N/A'),
                **run.data.metrics
            }
            run_data.append(run_info)
            
            logging.info(f"\nRun: {run.info.run_name} (ID: {run.info.run_id[:8]}...)")
            logging.info(f"  Status: {run.info.status}")
            logging.info(f"  Model: {run.data.tags.get('model_type', 'N/A')}")
            logging.info(f"  Start Time: {datetime.fromtimestamp(run.info.start_time/1000)}")
            
            # Show top metrics
            metrics = {k: v for k, v in run.data.metrics.items() 
                      if k in ['f1_score', 'roc_auc', 'accuracy']}
            for k, v in metrics.items():
                logging.info(f"  {k}: {v:.4f}")
        
        # Create DataFrame
        runs_df = pd.DataFrame(run_data)
        if not runs_df.empty:
            runs_df.to_csv('reports/mlflow/all_runs.csv', index=False)
            logging.info("\n✅ Runs data saved to reports/mlflow/all_runs.csv")
        
        return runs_df
    
    def compare_runs(self):
        """
        Compare all runs and identify best performing runs
        """
        logging.info("\n=== COMPARING MLFLOW RUNS ===")
        
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["metrics.f1_score DESC"]
        )
        
        if not runs:
            logging.warning("No runs found for comparison")
            return None
        
        # Collect run metrics for comparison
        comparison_data = []
        for run in runs:
            run_dict = {
                'run_id': run.info.run_id[:8],
                'run_name': run.info.run_name,
                'model_type': run.data.tags.get('model_type', 'N/A'),
                'f1_score': run.data.metrics.get('f1_score', 0),
                'roc_auc': run.data.metrics.get('roc_auc', 0),
                'pr_auc': run.data.metrics.get('pr_auc', 0),
                'accuracy': run.data.metrics.get('accuracy', 0),
                'precision': run.data.metrics.get('precision', 0),
                'recall': run.data.metrics.get('recall', 0),
                'cv_f1_mean': run.data.metrics.get('cv_f1_mean', 0),
                'status': run.info.status
            }
            comparison_data.append(run_dict)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display top runs
        logging.info("\nTop 5 runs by F1 Score:")
        top_runs = comparison_df.head(5)
        for idx, run in top_runs.iterrows():
            logging.info(f"\n  {idx+1}. {run['run_name']} ({run['model_type']})")
            logging.info(f"     F1: {run['f1_score']:.4f}, ROC-AUC: {run['roc_auc']:.4f}")
        
        # Save comparison
        comparison_df.to_csv('reports/mlflow/run_comparison.csv', index=False)
        logging.info("\n✅ Run comparison saved to reports/mlflow/run_comparison.csv")
        
        # Plot comparison
        self.plot_run_comparison(comparison_df)
        
        return comparison_df
    
    def plot_run_comparison(self, comparison_df):
        """
        Plot comparison of runs
        """
        plt.figure(figsize=(14, 8))
        
        # Get top 10 runs by F1
        top_10 = comparison_df.head(10).copy()
        
        # Create grouped bar chart
        metrics = ['f1_score', 'roc_auc', 'accuracy']
        x = np.arange(len(top_10))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, top_10[metric].values, width, 
                   label=metric, alpha=0.8)
        
        plt.xlabel('Run')
        plt.ylabel('Score')
        plt.title('Top 10 MLflow Runs - Performance Comparison')
        plt.xticks(x + width, [f"{row['run_name']}\n({row['model_type']})" 
                               for _, row in top_10.iterrows()], 
                  rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('reports/mlflow/run_comparison.png', dpi=100)
        plt.close()
        
        logging.info("✅ Run comparison plot saved to reports/mlflow/run_comparison.png")
    
    def get_best_run(self, metric='f1_score'):
        """
        Get the best run based on specified metric
        """
        logging.info(f"\n=== BEST RUN BY {metric.upper()} ===")
        
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if not runs:
            logging.warning("No runs found")
            return None
        
        best_run = runs[0]
        
        logging.info(f"Best Run: {best_run.info.run_name}")
        logging.info(f"Run ID: {best_run.info.run_id}")
        logging.info(f"Model Type: {best_run.data.tags.get('model_type', 'N/A')}")
        logging.info(f"Status: {best_run.info.status}")
        
        logging.info("\nMetrics:")
        for k, v in sorted(best_run.data.metrics.items()):
            if isinstance(v, (int, float)):
                logging.info(f"  {k}: {v:.4f}")
        
        # Save best run info
        best_run_info = {
            'run_id': best_run.info.run_id,
            'run_name': best_run.info.run_name,
            'model_type': best_run.data.tags.get('model_type', 'N/A'),
            'metrics': best_run.data.metrics,
            'params': best_run.data.params,
            'tags': best_run.data.tags
        }
        
        with open('reports/mlflow/best_run.json', 'w', encoding='utf-8') as f:
            json.dump(best_run_info, f, indent=2, default=str)
        
        logging.info("\n✅ Best run info saved to reports/mlflow/best_run.json")
        
        return best_run
    
    def register_best_model(self, model_name="insomnia_best_model"):
        """
        Register the best model in MLflow Model Registry
        """
        logging.info(f"\n=== REGISTERING BEST MODEL AS: {model_name} ===")
        
        # Get best run
        best_run = self.get_best_run()
        
        if not best_run:
            logging.error("No best run found")
            return None
        
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        
        try:
            # Register model
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            logging.info(f"✅ Model registered successfully!")
            logging.info(f"  Name: {registered_model.name}")
            logging.info(f"  Version: {registered_model.version}")
            logging.info(f"  Stage: {registered_model.current_stage}")
            
            # Add model description
            self.client.update_registered_model(
                name=model_name,
                description=f"Best Insomnia Risk Prediction Model - F1: {best_run.data.metrics.get('f1_score', 0):.4f}"
            )
            
            # Add run description
            self.client.update_model_version(
                name=model_name,
                version=registered_model.version,
                description=f"Run: {best_run.info.run_name}, Model: {best_run.data.tags.get('model_type', 'N/A')}"
            )
            
            # Transition to Production if it's the best
            self.client.transition_model_version_stage(
                name=model_name,
                version=registered_model.version,
                stage="Production"
            )
            logging.info(f"✅ Model moved to Production stage")
            
            return registered_model
            
        except Exception as e:
            logging.error(f"❌ Error registering model: {str(e)}")
            return None
    
    def export_experiment_summary(self):
        """
        Export complete experiment summary
        """
        logging.info("\n=== EXPORTING EXPERIMENT SUMMARY ===")
        
        summary = {
            'experiment_name': self.experiment_name,
            'experiment_id': self.experiment_id,
            'tracking_uri': self.tracking_uri,
            'export_date': datetime.now().isoformat(),
            'total_runs': 0,
            'models_trained': [],
            'best_run': None,
            'runs': []
        }
        
        runs = self.client.search_runs(experiment_ids=[self.experiment_id])
        summary['total_runs'] = len(runs)
        
        for run in runs:
            run_summary = {
                'run_id': run.info.run_id,
                'run_name': run.info.run_name,
                'status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time/1000).isoformat(),
                'model_type': run.data.tags.get('model_type', 'N/A'),
                'metrics': run.data.metrics,
                'params': run.data.params
            }
            summary['runs'].append(run_summary)
            
            if run.data.tags.get('model_type', 'N/A') not in summary['models_trained']:
                summary['models_trained'].append(run.data.tags.get('model_type', 'N/A'))
        
        # Get best run
        best_run = self.get_best_run()
        if best_run:
            summary['best_run'] = {
                'run_id': best_run.info.run_id,
                'run_name': best_run.info.run_name,
                'model_type': best_run.data.tags.get('model_type', 'N/A'),
                'f1_score': best_run.data.metrics.get('f1_score', 0)
            }
        
        # Save summary
        summary_path = 'reports/mlflow/experiment_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"✅ Experiment summary saved to {summary_path}")
        logging.info(f"  Total Runs: {summary['total_runs']}")
        logging.info(f"  Models Trained: {', '.join(summary['models_trained'])}")
        
        return summary
    
    def cleanup_old_runs(self, days=30):
        """
        Delete runs older than specified days
        """
        logging.info(f"\n=== CLEANING UP RUNS OLDER THAN {days} DAYS ===")
        
        cutoff_time = datetime.now().timestamp() * 1000 - (days * 24 * 60 * 60 * 1000)
        
        runs = self.client.search_runs(experiment_ids=[self.experiment_id])
        deleted_count = 0
        
        for run in runs:
            if run.info.start_time < cutoff_time:
                self.client.delete_run(run.info.run_id)
                deleted_count += 1
                logging.info(f"Deleted run: {run.info.run_name} ({run.info.run_id})")
        
        logging.info(f"✅ Deleted {deleted_count} old runs")
        
        return deleted_count
    
    def print_ui_instructions(self):
        """
        Print comprehensive MLflow UI instructions - FIXED for Windows encoding
        """
        instructions = """
============================================================
              MLFLOW UI INSTRUCTIONS
============================================================

1. START MLFLOW UI:
   -----------------
   Method 1 - From Python:
      python code/04_mlflow_tracking.py --start-ui
   
   Method 2 - Command Line:
      mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db
   
   Method 3 - Using this script:
      from 04_mlflow_tracking import MLflowManager
      manager = MLflowManager()
      manager.start_mlflow_ui()

2. ACCESS THE UI:
   ----------------
   * Open your browser and go to: http://localhost:5000
   * Default port is 5000, change with --port if needed

3. UI NAVIGATION:
   ---------------
   * EXPERIMENTS: View all experiments in the left sidebar
   * RUNS: Click on experiment to see all runs
   * METRICS: Compare metrics across runs with parallel coordinates
   * PARAMETERS: View and compare hyperparameters
   * ARTIFACTS: Download models, plots, and other artifacts
   * REGISTRY: Manage registered models (Models tab)

4. KEY FEATURES:
   --------------
   * Compare runs: Select runs and click "Compare"
   * Search runs: Use search bar to filter runs
   * Download artifacts: Click on artifact to download
   * Register models: Promote best runs to Model Registry
   * Share results: Export runs as CSV or share experiment link

5. TROUBLESHOOTING:
   -----------------
   * Port in use: Try different port: mlflow ui --port 5001
   * DB locked: Close other MLflow instances
   * No runs: Check backend-store-uri matches your mlflow.db location

6. USEFUL COMMANDS:
   -----------------
   * List experiments: mlflow experiments list
   * Delete experiment: mlflow experiments delete --experiment-id <ID>
   * Download artifacts: mlflow artifacts download --run-id <ID>
   * Serve model: mlflow models serve -m runs:/<RUN_ID>/model -p 1234

7. MODEL SERVING:
   ---------------
   Once you have a registered model, serve it with:
      mlflow models serve -m "models:/insomnia_best_model/Production" -p 1234
   
   Test with:
      curl -X POST -H "Content-Type: application/json" ^
           -d "{\"data\": [<your_features>]}" ^
           http://localhost:1234/invocations

For more details: https://www.mlflow.org/docs/latest/tracking.html
"""
        # Print to console
        print(instructions)
        
        # Save instructions to file with UTF-8 encoding
        with open('reports/mlflow/mlflow_ui_instructions.txt', 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        logging.info("✅ UI instructions saved to reports/mlflow/mlflow_ui_instructions.txt")
    
    def run_tracking_analysis(self):
        """
        Run complete MLflow tracking analysis
        """
        logging.info("="*60)
        logging.info("STARTING MLFLOW TRACKING ANALYSIS")
        logging.info("="*60)
        
        try:
            # Step 1: Setup experiment
            self.setup_experiment()
            
            # Step 2: List all experiments
            self.list_experiments()
            
            # Step 3: Get experiment runs
            runs_df = self.get_experiment_runs()
            
            # Step 4: Compare runs
            if runs_df is not None and not runs_df.empty:
                self.compare_runs()
            
            # Step 5: Get best run
            best_run = self.get_best_run()
            
            # Step 6: Export summary
            self.export_experiment_summary()
            
            # Step 7: Print UI instructions
            self.print_ui_instructions()
            
            # Step 8: Check if UI is running
            self.check_mlflow_server()
            
            logging.info("="*60)
            logging.info("✅ MLFLOW TRACKING ANALYSIS COMPLETED")
            logging.info("="*60)
            
            return best_run
            
        except Exception as e:
            logging.error(f"❌ Error in tracking analysis: {str(e)}")
            return None


def main():
    """
    Main function to run MLflow tracking
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='MLflow Tracking Manager')
    parser.add_argument('--start-ui', action='store_true', 
                       help='Start MLflow UI')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port for MLflow UI (default: 5000)')
    parser.add_argument('--analyze', action='store_true',
                       help='Run tracking analysis')
    parser.add_argument('--register-best', action='store_true',
                       help='Register best model')
    parser.add_argument('--cleanup', type=int, default=0,
                       help='Delete runs older than specified days')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = MLflowManager()
    
    if args.start_ui:
        manager.start_mlflow_ui(port=args.port)
        
    elif args.analyze:
        manager.run_tracking_analysis()
        
    elif args.register_best:
        manager.setup_experiment()
        manager.register_best_model()
        
    elif args.cleanup > 0:
        manager.setup_experiment()
        manager.cleanup_old_runs(days=args.cleanup)
        
    else:
        # Default: print UI instructions
        manager.print_ui_instructions()
        
        # Check if UI is running
        if not manager.check_mlflow_server():
            response = input("\nStart MLflow UI? (y/n): ")
            if response.lower() == 'y':
                manager.start_mlflow_ui()


if __name__ == "__main__":
    main()