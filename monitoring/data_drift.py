"""
Data Drift Detection - Monitors input feature changes
"""

import numpy as np
import pandas as pd
from scipy import stats
import json

class DataDriftDetector:
    """Detect data drift using KS Test and PSI"""
    
    def __init__(self, baseline_stats_path='../model/baseline_stats.json'):
        with open(baseline_stats_path, 'r') as f:
            self.baseline_stats = json.load(f)
        self.feature_names = self.baseline_stats['feature_names'][:7]  # Only numeric features
    
    def calculate_psi(self, baseline_array, current_array, bins=10):
        """
        Population Stability Index
        PSI < 0.1: No drift
        0.1-0.2: Moderate drift
        > 0.2: Severe drift
        """
        breakpoints = np.percentile(baseline_array, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        if len(breakpoints) <= 1:
            return 0.0
        
        baseline_percents = np.histogram(baseline_array, bins=breakpoints)[0] / len(baseline_array)
        current_percents = np.histogram(current_array, bins=breakpoints)[0] / len(current_array)
        
        baseline_percents = np.where(baseline_percents == 0, 0.0001, baseline_percents)
        current_percents = np.where(current_percents == 0, 0.0001, current_percents)
        
        psi = np.sum((current_percents - baseline_percents) * np.log(current_percents / baseline_percents))
        
        return psi
    
    def ks_test(self, baseline_array, current_array):
        """Kolmogorov-Smirnov Test"""
        statistic, p_value = stats.ks_2samp(baseline_array, current_array)
        return statistic, p_value
    
    def analyze_feature_drift(self, current_df, baseline_df):
        """Analyze drift for all features"""
        drift_results = {}
        
        for feature in self.feature_names:
            baseline_array = baseline_df[feature].values
            current_array = current_df[feature].values
            
            # PSI
            psi_value = self.calculate_psi(baseline_array, current_array)
            
            # KS Test
            ks_stat, ks_pval = self.ks_test(baseline_array, current_array)
            
            # Statistics
            baseline_mean = baseline_array.mean()
            current_mean = current_array.mean()
            mean_shift = ((current_mean - baseline_mean) / baseline_mean) * 100
            
            # Severity
            if psi_value < 0.1 and ks_pval > 0.05:
                severity = "No Drift"
                alert = "✅"
            elif psi_value < 0.2 and ks_pval > 0.01:
                severity = "Moderate Drift"
                alert = "⚠️"
            else:
                severity = "Severe Drift"
                alert = "🚨"
            
            drift_results[feature] = {
                'psi': round(psi_value, 4),
                'ks_statistic': round(ks_stat, 4),
                'ks_pvalue': round(ks_pval, 4),
                'baseline_mean': round(baseline_mean, 2),
                'current_mean': round(current_mean, 2),
                'mean_shift_pct': round(mean_shift, 2),
                'severity': severity,
                'alert': alert
            }
        
        return drift_results
    
    def generate_report(self, drift_results):
        """Generate readable report"""
        report = []
        report.append("=" * 80)
        report.append("DATA DRIFT DETECTION REPORT")
        report.append("=" * 80)
        
        severe = sum(1 for r in drift_results.values() if r['severity'] == 'Severe Drift')
        moderate = sum(1 for r in drift_results.values() if r['severity'] == 'Moderate Drift')
        none = sum(1 for r in drift_results.values() if r['severity'] == 'No Drift')
        
        report.append(f"\n📊 Summary:")
        report.append(f"   🚨 Severe Drift: {severe} features")
        report.append(f"   ⚠️  Moderate Drift: {moderate} features")
        report.append(f"   ✅ No Drift: {none} features")
        
        report.append(f"\n📈 Feature Analysis:")
        report.append("-" * 80)
        
        for feature, metrics in drift_results.items():
            report.append(f"\n{metrics['alert']} {feature.upper()} - {metrics['severity']}")
            report.append(f"   PSI Score: {metrics['psi']:.4f}")
            report.append(f"   KS Test: stat={metrics['ks_statistic']:.4f}, p={metrics['ks_pvalue']:.4f}")
            report.append(f"   Mean: {metrics['baseline_mean']:.2f} → {metrics['current_mean']:.2f} ({metrics['mean_shift_pct']:+.1f}%)")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def get_drift_score(self, drift_results):
        """Overall drift score (0-100)"""
        psi_scores = [r['psi'] for r in drift_results.values()]
        avg_psi = np.mean(psi_scores)
        drift_score = min(avg_psi * 100, 100)
        return round(drift_score, 2)

# Test the detector
if __name__ == "__main__":
    print("🔍 Testing Data Drift Detection")
    print("=" * 80)
    
    # Load data
    baseline_df = pd.read_csv('../data/train.csv')
    batch_1 = pd.read_csv('../data/batch_1_normal.csv')
    batch_2 = pd.read_csv('../data/batch_2_flash_sale.csv')
    batch_3 = pd.read_csv('../data/batch_3_supply_crisis.csv')
    
    detector = DataDriftDetector()
    
    batches = [
        ("Batch 1 (Normal)", batch_1),
        ("Batch 2 (Flash Sale)", batch_2),
        ("Batch 3 (Supply Crisis)", batch_3)
    ]
    
    for batch_name, batch_df in batches:
        print(f"\n\n{'=' * 80}")
        print(f"ANALYZING: {batch_name}")
        print('=' * 80)
        
        drift_results = detector.analyze_feature_drift(batch_df, baseline_df)
        report = detector.generate_report(drift_results)
        print(report)
        
        drift_score = detector.get_drift_score(drift_results)
        print(f"\n🎯 Overall Drift Score: {drift_score}/100")
        
        if drift_score < 10:
            print("   Status: ✅ HEALTHY")
        elif drift_score < 20:
            print("   Status: ⚠️  WARNING")
        else:
            print("   Status: 🚨 CRITICAL - Retraining recommended")