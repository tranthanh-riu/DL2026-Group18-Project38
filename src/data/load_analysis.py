import os
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


def calculate_rms(X):
    return np.sqrt(np.mean(X ** 2, axis=1))


def calculate_fft(X, sampling_rate=12000):
    n = X.shape[1]

    fft_values = np.fft.rfft(X, axis=1)
    magnitude = np.abs(fft_values) / n

    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

    return frequencies, magnitude


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_stats = []

    for load in range(4):
        path = os.path.join(PROCESSED_DIR, f"load{load}_test.npz")

        data = np.load(path, allow_pickle=True)
        X = data["X"]
        y = data["y"]

        normal_X = X[y == 0]

        # RMS
        rms = calculate_rms(normal_X)

        # FFT
        frequencies, magnitude = calculate_fft(normal_X)

        mean_rms = float(np.mean(rms))
        std_rms = float(np.std(rms))

        mean_fft = np.mean(magnitude, axis=0)

        peak_index = int(np.argmax(mean_fft))
        peak_frequency = float(frequencies[peak_index])
        peak_magnitude = float(mean_fft[peak_index])

        all_stats.append(
            [
                load,
                len(normal_X),
                mean_rms,
                std_rms,
                peak_frequency,
                peak_magnitude,
            ]
        )

        np.savez(
            os.path.join(RESULTS_DIR, f"load{load}_rms_fft.npz"),
            rms=rms,
            frequencies=frequencies,
            mean_fft=mean_fft,
        )

        print(
            f"Load {load}: "
            f"normal_windows={len(normal_X)}, "
            f"RMS_mean={mean_rms:.10f}, "
            f"RMS_std={std_rms:.10f}, "
            f"FFT_peak_frequency={peak_frequency:.2f} Hz, "
            f"FFT_peak_magnitude={peak_magnitude:.10f}"
        )

    summary_path = os.path.join(RESULTS_DIR, "load_rms_fft_summary.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(
            "load,normal_windows,rms_mean,rms_std,"
            "fft_peak_frequency_hz,fft_peak_magnitude\n"
        )

        for row in all_stats:
            f.write(
                f"{row[0]},{row[1]},{row[2]:.10f},"
                f"{row[3]:.10f},{row[4]:.2f},{row[5]:.10f}\n"
            )

    print()
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()