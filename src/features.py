import numpy as np
from scipy.stats import kurtosis, skew


def extract_features(X):
    # X: (N, 1024) đã chuẩn hóa. Mỗi window -> 8 đặc trưng thống kê (miền thời gian)
    rms = np.sqrt((X ** 2).mean(axis=1))
    std = X.std(axis=1)
    mean_abs = np.abs(X).mean(axis=1)
    peak = np.abs(X).max(axis=1)
    peak_to_peak = X.max(axis=1) - X.min(axis=1)
    # Crest factor = peak / RMS; cộng số rất nhỏ để tránh chia cho 0
    crest_factor = peak / (rms + 1e-12)
    skewness = skew(X, axis=1)
    kurt = kurtosis(X, axis=1)
    # Ghép thành (N, 8): mỗi cột là một đặc trưng
    return np.stack([rms, std, mean_abs, peak, peak_to_peak, crest_factor, skewness, kurt], axis=1)


def main():
    # Test shape bằng dữ liệu ngẫu nhiên
    X = np.random.randn(5, 1024).astype(np.float32)
    features = extract_features(X)
    print("features.shape =", features.shape, "(ky vong (5, 8))")


if __name__ == "__main__":
    main()
