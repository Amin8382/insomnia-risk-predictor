"""
02_feature_selection.py
Feature Selection for Insomnia Risk Predictor
Multiple methods: correlation analysis, mutual information, feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import logging
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_selection.log'),
        logging.StreamHandler()
    ]
)

class FeatureSelector:
    """
    Comprehensive feature selection for insomnia prediction
    """
    
    def __init__(self, data_path='data/processed/train.csv'):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.selected_features = None
        self.correlation_matrix = None
        
        # Create directories
        os.makedirs('reports/figures', exist_ok=True)
        os.makedirs('models/features', exist_ok=True)
        
    def load_processed_data(self):
        """
        Load the preprocessed training data
        """
        logging.info(f"Loading processed data from {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        logging.info(f"Data shape: {self.df.shape}")
        
        # Separate features and target
        self.y = self.df['insomnia_class']
        self.X = self.df.drop('insomnia_class', axis=1)
        self.feature_names = list(self.X.columns)
        
        logging.info(f"Features: {len(self.feature_names)}")
        logging.info(f"Target distribution:\n{self.y.value_counts(normalize=True)}")
        
        return self.X, self.y
    
    def analyze_correlations(self, threshold=0.1):
        """
        Analyze correlations with target and between features
        """
        logging.info("\n=== CORRELATION ANALYSIS ===")
        
        # Calculate correlations
        df_with_target = self.df.copy()
        correlations = df_with_target.corr()['insomnia_class'].drop('insomnia_class')
        
        # Sort by absolute correlation
        correlations_abs = correlations.abs().sort_values(ascending=False)
        
        logging.info("\nTop 10 features correlated with insomnia:")
        for feature in correlations_abs.head(10).index:
            corr = correlations[feature]
            logging.info(f"  {feature}: {corr:.4f}")
        
        # Features with correlation above threshold
        high_corr_features = correlations_abs[correlations_abs > threshold].index.tolist()
        logging.info(f"\nFeatures with |correlation| > {threshold}: {len(high_corr_features)}")
        
        # Plot top correlations
        plt.figure(figsize=(10, 8))
        top_features = correlations_abs.head(15).index
        top_corrs = correlations[top_features]
        
        colors = ['red' if x < 0 else 'green' for x in top_corrs]
        plt.barh(range(len(top_corrs)), top_corrs.values, color=colors)
        plt.yticks(range(len(top_corrs)), top_corrs.index)
        plt.xlabel('Correlation with Insomnia')
        plt.title('Top 15 Features Correlated with Insomnia')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/figures/feature_correlations.png', dpi=100)
        plt.close()
        
        logging.info("Correlation plot saved to reports/figures/feature_correlations.png")
        
        # Check for multicollinearity
        correlation_matrix = df_with_target[self.feature_names].corr()
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find highly correlated features
        high_corr_pairs = []
        for column in upper_tri.columns:
            high_corr = upper_tri[column][abs(upper_tri[column]) > 0.8]
            if len(high_corr) > 0:
                for idx, value in high_corr.items():
                    high_corr_pairs.append((column, idx, value))
        
        if high_corr_pairs:
            logging.info("\nHighly correlated feature pairs (|r| > 0.8):")
            for f1, f2, corr in high_corr_pairs:
                logging.info(f"  {f1} - {f2}: {corr:.4f}")
        
        return correlations, high_corr_features
    
    def mutual_information_selection(self, k=20):
        """
        Select top k features using mutual information
        """
        logging.info("\n=== MUTUAL INFORMATION SELECTION ===")
        
        # Calculate mutual information
        mi_selector = SelectKBest(mutual_info_classif, k='all')
        mi_selector.fit(self.X, self.y)
        
        # Get scores
        mi_scores = pd.DataFrame({
            'feature': self.feature_names,
            'mi_score': mi_selector.scores_
        }).sort_values('mi_score', ascending=False)
        
        logging.info("\nTop 10 features by Mutual Information:")
        for i, row in mi_scores.head(10).iterrows():
            logging.info(f"  {row['feature']}: {row['mi_score']:.4f}")
        
        # Plot MI scores
        plt.figure(figsize=(10, 8))
        top_mi = mi_scores.head(15)
        
        plt.barh(range(len(top_mi)), top_mi['mi_score'].values)
        plt.yticks(range(len(top_mi)), top_mi['feature'].values)
        plt.xlabel('Mutual Information Score')
        plt.title('Top 15 Features by Mutual Information')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/figures/mutual_information.png', dpi=100)
        plt.close()
        
        logging.info("MI plot saved to reports/figures/mutual_information.png")
        
        # Select top k features
        top_k_features = mi_scores.head(k)['feature'].tolist()
        logging.info(f"\nSelected top {k} features by MI")
        
        return mi_scores, top_k_features
    
    def rfe_selection(self, n_features=15):
        """
        Recursive Feature Elimination with Logistic Regression
        """
        logging.info("\n=== RECURSIVE FEATURE ELIMINATION ===")
        
        # Base model for RFE
        model = LogisticRegression(max_iter=1000, random_state=42)
        
        # RFE selector
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        rfe.fit(self.X, self.y)
        
        # Get selected features
        selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) 
                           if rfe.support_[i]]
        
        # Get rankings
        rankings = pd.DataFrame({
            'feature': self.feature_names,
            'rfe_rank': rfe.ranking_
        }).sort_values('rfe_rank')
        
        logging.info(f"\nTop {n_features} features selected by RFE:")
        for feature in selected_features:
            logging.info(f"  {feature}")
        
        return rankings, selected_features
    
    def random_forest_importance(self, n_features=15):
        """
        Feature importance from Random Forest
        """
        logging.info("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
        
        # Train a quick Random Forest for importance
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X, self.y)
        
        # Get feature importances
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info("\nTop 10 features by Random Forest importance:")
        for i, row in importances.head(10).iterrows():
            logging.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Plot importances
        plt.figure(figsize=(10, 8))
        top_imp = importances.head(15)
        
        plt.barh(range(len(top_imp)), top_imp['importance'].values)
        plt.yticks(range(len(top_imp)), top_imp['feature'].values)
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Features by Random Forest Importance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/figures/rf_importance.png', dpi=100)
        plt.close()
        
        logging.info("RF importance plot saved to reports/figures/rf_importance.png")
        
        # Select top n features
        top_features = importances.head(n_features)['feature'].tolist()
        
        return importances, top_features
    
    def statistical_tests(self):
        """
        Perform statistical tests for each feature
        """
        logging.info("\n=== STATISTICAL TESTS ===")
        
        results = []
        classes = self.y.unique()
        
        for feature in self.feature_names:
            # Separate by class
            class_0 = self.X[self.y == 0][feature]
            class_1 = self.X[self.y == 1][feature]
            
            # T-test
            t_stat, p_value = stats.ttest_ind(class_0, class_1)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((class_0.std()**2 + class_1.std()**2) / 2)
            cohens_d = (class_1.mean() - class_0.mean()) / pooled_std if pooled_std != 0 else 0
            
            results.append({
                'feature': feature,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('cohens_d', ascending=False)
        
        logging.info("\nFeatures with largest effect size (Cohen's d):")
        for i, row in results_df.head(10).iterrows():
            logging.info(f"  {row['feature']}: d={row['cohens_d']:.4f}, "
                        f"p={row['p_value']:.4f} {'*' if row['significant'] else ''}")
        
        return results_df
    
    def combine_selections(self, methods=['correlation', 'mi', 'rfe', 'rf'], 
                          top_k_each=15):
        """
        Combine results from multiple selection methods
        """
        logging.info("\n=== COMBINING SELECTION METHODS ===")
        
        all_selections = []
        
        # Get selections from each method
        if 'correlation' in methods:
            corr, _ = self.analyze_correlations()
            corr_top = corr.abs().sort_values(ascending=False).head(top_k_each).index.tolist()
            all_selections.extend(corr_top)
            logging.info(f"Correlation: {len(corr_top)} features")
        
        if 'mi' in methods:
            mi_scores, mi_top = self.mutual_information_selection(k=top_k_each)
            all_selections.extend(mi_top)
            logging.info(f"Mutual Info: {len(mi_top)} features")
        
        if 'rfe' in methods:
            _, rfe_top = self.rfe_selection(n_features=top_k_each)
            all_selections.extend(rfe_top)
            logging.info(f"RFE: {len(rfe_top)} features")
        
        if 'rf' in methods:
            _, rf_top = self.random_forest_importance(n_features=top_k_each)
            all_selections.extend(rf_top)
            logging.info(f"Random Forest: {len(rf_top)} features")
        
        # Count frequencies
        from collections import Counter
        feature_counts = Counter(all_selections)
        
        # Get features selected by at least 2 methods
        consensus_features = [f for f, count in feature_counts.items() 
                            if count >= 2]
        
        # Get features selected by all methods
        all_methods_features = [f for f, count in feature_counts.items() 
                              if count == len(methods)]
        
        logging.info(f"\nConsensus features (≥2 methods): {len(consensus_features)}")
        logging.info(f"Features selected by all methods: {len(all_methods_features)}")
        
        if all_methods_features:
            logging.info("\nFeatures selected by ALL methods:")
            for f in all_methods_features:
                logging.info(f"  {f}")
        
        # Save feature importance summary
        summary_df = pd.DataFrame({
            'feature': list(feature_counts.keys()),
            'selection_count': list(feature_counts.values())
        }).sort_values('selection_count', ascending=False)
        
        summary_df.to_csv('models/features/feature_selection_summary.csv', index=False)
        logging.info("\nFeature selection summary saved to models/features/feature_selection_summary.csv")
        
        # Plot consensus
        plt.figure(figsize=(12, 8))
        top_consensus = summary_df.head(20)
        
        plt.barh(range(len(top_consensus)), top_consensus['selection_count'].values)
        plt.yticks(range(len(top_consensus)), top_consensus['feature'].values)
        plt.xlabel('Number of Methods Selecting Feature')
        plt.title('Feature Consensus Across Selection Methods')
        plt.xticks(range(1, len(methods)+1))
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/figures/feature_consensus.png', dpi=100)
        plt.close()
        
        return consensus_features, all_methods_features, summary_df
    
    def select_final_features(self, method='consensus', min_count=2):
        """
        Select final feature set based on chosen method
        """
        logging.info(f"\n=== SELECTING FINAL FEATURES (method: {method}) ===")
        
        if method == 'consensus':
            # Get consensus features (selected by at least min_count methods)
            consensus, _, summary = self.combine_selections()
            self.selected_features = [f for f in consensus 
                                     if f in self.feature_names]
            
        elif method == 'correlation':
            corr, _ = self.analyze_correlations()
            self.selected_features = corr.abs().sort_values(
                ascending=False
            ).head(20).index.tolist()
            
        elif method == 'mi':
            _, self.selected_features = self.mutual_information_selection(k=20)
            
        elif method == 'rfe':
            _, self.selected_features = self.rfe_selection(n_features=20)
            
        elif method == 'rf':
            _, self.selected_features = self.random_forest_importance(n_features=20)
        
        logging.info(f"\nFinal selected features ({len(self.selected_features)}):")
        for i, f in enumerate(self.selected_features, 1):
            logging.info(f"  {i}. {f}")
        
        # Save selected features
        with open('models/features/selected_features.txt', 'w') as f:
            for feature in self.selected_features:
                f.write(f"{feature}\n")
        
        # Create reduced dataset
        X_selected = self.X[self.selected_features]
        
        # Save reduced dataset
        reduced_df = X_selected.copy()
        reduced_df['insomnia_class'] = self.y
        reduced_df.to_csv('data/processed/train_selected.csv', index=False)
        
        logging.info("\n✅ Reduced dataset saved to data/processed/train_selected.csv")
        
        return self.selected_features, X_selected
    
    def generate_report(self):
        """
        Generate a comprehensive feature selection report
        """
        logging.info("\n=== GENERATING FEATURE SELECTION REPORT ===")
        
        report = []
        report.append("="*60)
        report.append("FEATURE SELECTION REPORT")
        report.append("="*60)
        report.append(f"Total features analyzed: {len(self.feature_names)}")
        report.append("")
        
        # Correlation analysis
        corr, _ = self.analyze_correlations()
        report.append("\nTOP 10 CORRELATIONS:")
        for f in corr.abs().sort_values(ascending=False).head(10).index:
            report.append(f"  {f}: {corr[f]:.4f}")
        
        # Statistical tests
        stats_df = self.statistical_tests()
        report.append("\nTOP 10 EFFECT SIZES (Cohen's d):")
        for _, row in stats_df.head(10).iterrows():
            report.append(f"  {row['feature']}: d={row['cohens_d']:.4f}, "
                         f"p={row['p_value']:.4f}")
        
        # Final selection
        report.append("\nFINAL SELECTED FEATURES:")
        for i, f in enumerate(self.selected_features, 1):
            report.append(f"  {i}. {f}")
        
        # Save report
        report_text = "\n".join(report)
        with open('reports/feature_selection_report.txt', 'w') as f:
            f.write(report_text)
        
        logging.info("Report saved to reports/feature_selection_report.txt")
        
        return report_text
    
    def run_selection_pipeline(self, method='consensus'):
        """
        Run the complete feature selection pipeline
        """
        logging.info("="*60)
        logging.info("STARTING FEATURE SELECTION PIPELINE")
        logging.info("="*60)
        
        # Step 1: Load data
        self.load_processed_data()
        
        # Step 2: Run all analyses
        self.analyze_correlations()
        self.mutual_information_selection()
        self.rfe_selection()
        self.random_forest_importance()
        self.statistical_tests()
        
        # Step 3: Select final features
        selected_features, X_selected = self.select_final_features(method=method)
        
        # Step 4: Generate report
        self.generate_report()
        
        logging.info("="*60)
        logging.info("✅ FEATURE SELECTION PIPELINE COMPLETED")
        logging.info("="*60)
        
        return selected_features, X_selected


# Execute feature selection
if __name__ == "__main__":
    # Initialize selector
    selector = FeatureSelector('data/processed/train.csv')
    
    # Run full pipeline
    # Options for method: 'consensus', 'correlation', 'mi', 'rfe', 'rf'
    selected_features, X_selected = selector.run_selection_pipeline(method='consensus')
    
    print("\n✅ Feature selection complete! Ready for model training.")