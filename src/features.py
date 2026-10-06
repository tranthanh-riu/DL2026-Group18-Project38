import numpy as np
from scipy.stats import skew, kurtosis

def extract_features(x):
    """
    Extracts statistical features from time-series windows.
    Args:
        x: numpy array of shape (n_windows, window_size)
    Returns:
        features: numpy array of shape (n_windows, 8)
    """
    # rms = sqrt(mean(x^2))
    rms = np.sqrt(np.mean(x**2, axis=1))
    
    # std
    std = np.std(x, axis=1)
    
    # mean abs
    mean_abs = np.mean(np.abs(x), axis=1)
    
    # peak = max(abs(x))
    peak = np.max(np.abs(x), axis=1)
    
    # peak-to-peak = max(x) - min(x)
    p2p = np.max(x, axis=1) - np.min(x, axis=1)
    
    # crest factor = peak / rms
    # Add a small epsilon to avoid division by zero
    crest_factor = peak / (rms + 1e-8)
    
    # skewness
    skewness = skew(x, axis=1)
    
    # kurtosis
    kurt = kurtosis(x, axis=1)
    
    # stack features
    features = np.column_stack([
        rms, std, mean_abs, peak, p2p, crest_factor, skewness, kurt
    ])
    
    return features
