"""Additive white Gaussian noise at a target SNR (Setup3 robustness, test data only).

Tags: 'snr20', 'snr10', 'snr5' (dB). The noise level is FIXED, like a sensor noise floor:
the SNR is defined relative to the power of the normalisation reference signal, which is
1 in normalised units (global norm: normal 0 HP train has mean 0, std 1). Every window
gets the same noise std, so the noise carries no information about the window's own
amplitude. (A per-window SNR would scale the noise with each window's energy, so faults,
which are louder, would get more noise and become artificially easier to detect.)
Noise is generated with a fixed RNG per (seed, SNR), so scores are reproducible.
"""
import numpy as np

REF_POWER = 1.0  # variance of the normalisation reference set, in normalised units


def parse_snr(noise: str) -> float:
    if not noise.startswith("snr"):
        raise ValueError(f"unknown noise tag '{noise}' (expected e.g. 'snr10')")
    return float(noise[3:])


def noise_std(snr_db: float) -> float:
    return float(np.sqrt(REF_POWER / 10 ** (snr_db / 10)))


def add_noise(X: np.ndarray, noise: str, seed: int) -> np.ndarray:
    """X: (N, 1, window), normalised. Returns a noisy float32 copy."""
    snr_db = parse_snr(noise)
    rng = np.random.default_rng([seed, int(round(snr_db * 10))])
    return (X + noise_std(snr_db) * rng.standard_normal(X.shape)).astype(np.float32)
