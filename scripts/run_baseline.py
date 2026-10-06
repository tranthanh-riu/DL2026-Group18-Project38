import os
import numpy as np
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features import extract_features
from src.models.iforest import AnomalyIsolationForest

def run_baseline(data_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    seeds = [0, 1, 2]
    loads = [0, 1, 2, 3]
    
    # Load training data (load0_train)
    train_data = np.load(f"{data_dir}/load0_train.npz")
    X_train = train_data['X']
    features_train = extract_features(X_train)
    
    # Also evaluate on load0_val to provide validation scores for thresholding
    val_data = np.load(f"{data_dir}/load0_val.npz")
    X_val = val_data['X']
    y_val = val_data['y']
    features_val = extract_features(X_val)
    
    for seed in seeds:
        print(f"Running IF for seed {seed}...")
        model = AnomalyIsolationForest(n_estimators=200, random_state=seed)
        model.fit(features_train)
        
        # Predict on validation set
        val_scores = model.predict_score(features_val)
        np.savez(f"{results_dir}/iforest_val_seed{seed}.npz", score=val_scores, y=y_val)
        
        # Predict on test sets
        for L in loads:
            test_data = np.load(f"{data_dir}/load{L}_test.npz")
            X_test = test_data['X']
            y_test = test_data['y']
            features_test = extract_features(X_test)
            
            test_scores = model.predict_score(features_test)
            
            # Save scores
            out_file = f"{results_dir}/iforest_load{L}_test_seed{seed}.npz"
            np.savez(out_file, score=test_scores, y=y_test)
            print(f"  Saved {out_file}")

if __name__ == "__main__":
    run_baseline("data/processed", "results/scores")
