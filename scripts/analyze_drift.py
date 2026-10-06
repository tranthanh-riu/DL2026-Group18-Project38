import os
import numpy as np
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import extract_features

def analyze_feature_drift(data_dir, output_file):
    loads = [0, 1, 2, 3]
    feature_names = ['rms', 'std', 'mean_abs', 'peak', 'p2p', 'crest_factor', 'skewness', 'kurtosis']
    
    results = []
    
    # We will analyze test data only, and only normal windows (y=0)
    for L in loads:
        data = np.load(f"{data_dir}/load{L}_test.npz")
        X = data['X']
        y = data['y']
        
        # Filter normal windows
        X_normal = X[y == 0]
        
        # Extract features
        features = extract_features(X_normal)
        
        # Calculate mean and std for each feature
        f_mean = np.mean(features, axis=0)
        f_std = np.std(features, axis=0)
        
        for i, f_name in enumerate(feature_names):
            results.append({
                'load': f'{L}HP',
                'feature': f_name,
                'mean': f_mean[i],
                'std': f_std[i]
            })
            
    df = pd.DataFrame(results)
    
    # Calculate difference from 0HP
    df_pivot_mean = df.pivot(index='feature', columns='load', values='mean')
    
    drift_analysis = []
    for f_name in feature_names:
        base_mean = df_pivot_mean.loc[f_name, '0HP']
        drifts = {}
        for L in [1, 2, 3]:
            mean_L = df_pivot_mean.loc[f_name, f'{L}HP']
            # Percent change
            if base_mean != 0:
                pct_change = (mean_L - base_mean) / abs(base_mean) * 100
            else:
                pct_change = float('inf')
            drifts[f'{L}HP_change_%'] = pct_change
            
        drift_analysis.append({
            'feature': f_name,
            '0HP_mean': base_mean,
            **drifts
        })
        
    df_drift = pd.DataFrame(drift_analysis)
    
    print("\nFeature Drift Analysis (Change relative to 0 HP):")
    print(df_drift.to_string(index=False))
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_drift.to_csv(output_file, index=False)
    print(f"\nSaved analysis to {output_file}")
    
    # Identify which features drift the most
    max_drifts = []
    for _, row in df_drift.iterrows():
        max_drift = max(abs(row['1HP_change_%']), abs(row['2HP_change_%']), abs(row['3HP_change_%']))
        max_drifts.append((row['feature'], max_drift))
        
    max_drifts.sort(key=lambda x: x[1], reverse=True)
    print(f"\nFeatures that drift the most and likely cause false positives for IF:")
    for f, d in max_drifts[:3]:
        print(f"- {f}: up to {d:.1f}% change")

if __name__ == "__main__":
    analyze_feature_drift("data/processed", "results/tables/feature_drift.csv")
