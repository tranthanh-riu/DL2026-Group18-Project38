import numpy as np


def add_noise(X_raw, snr_db, seed):
    # Thêm nhiễu Gaussian vào tín hiệu THÔ (trước chuẩn hóa) với SNR cho trước; X_raw: (N, 1024)
    # P = công suất từng window (N, 1); sigma = sqrt(P / 10^(SNR/10))
    power = (X_raw ** 2).mean(axis=1, keepdims=True)
    sigma = np.sqrt(power / (10 ** (snr_db / 10)))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(X_raw.shape) * sigma
    return (X_raw + noise).astype(np.float32)


def main():
    # Kiểm tra: SNR đo lại từ nhiễu phải gần giá trị yêu cầu
    X_raw = np.random.randn(200, 1024).astype(np.float32)
    for snr_db in [20, 10, 5]:
        noisy = add_noise(X_raw, snr_db, seed=0)
        measured = 10 * np.log10((X_raw ** 2).mean() / ((noisy - X_raw) ** 2).mean())
        print(f"SNR yeu cau {snr_db} dB -> do lai {measured:.2f} dB")


if __name__ == "__main__":
    main()
