# DATA.md — CWRU Bearing Dataset

## 1. Source

| Item | Value |
|---|---|
| Dataset | Case Western Reserve University (CWRU) Bearing Data Center |
| Official URL | https://engineering.case.edu/bearingdatacenter/download-data-file |
| File URL pattern | `https://engineering.case.edu/sites/default/files/{file_id}.mat` |
| Version | CWRU publishes no version number. The version used here is identified by the **download date (2026-10-06 UTC)** and the **SHA-256 checksums** in Section 6 (also written to `data/raw/SHA256SUMS`). |
| Reference | Smith, W. A. & Randall, R. B. (2015). Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study. *Mechanical Systems and Signal Processing*. |

Test rig: 2 hp induction motor; accelerometers at the drive end and fan end; single-point faults introduced by electro-discharge machining; data recorded at motor loads 0–3 hp.

## 2. Files used (16 recordings)

Only the **drive-end (DE) accelerometer** signal is used. All 16 files are **48 kHz** recordings.

| Load | Normal | Inner race 0.007" | Ball 0.007" | Outer race @6:00 0.007" | Motor speed (rpm, from file) |
|---|---|---|---|---|---|
| 0 HP | 97 | 109 | 122 | 135 | 1796 |
| 1 HP | 98 | 110 | 123 | 136 | 1772 |
| 2 HP | 99 | 111 | 124 | 137 | 1748–1750 |
| 3 HP | 100 | 112 | 125 | 138 | 1721–1725 |

Full list with signal variable names: `data/manifest.csv`.

### Why 48 kHz fault files (not the 12 kHz set)

The normal baseline recordings (97–100) are 48 kHz. Two checks confirm this:

1. **Duration.** File 97 has 243,938 samples, exactly the same as the 48 kHz fault file 109 (0 HP, 5 s at 48 kHz). The 12 kHz fault files have about 121k samples for 10 s.
2. **Spectrum.** Read at 48 kHz, file 97 shows the same spectral peaks (360 Hz, 658 Hz) as the 48 kHz ball-fault file 122. The 12 kHz inner-race file 105 read at 12 kHz matches the 48 kHz file 109 read at 48 kHz (peaks near 455, 617, 677 Hz).

Mixing 48 kHz normal data with 12 kHz fault data would make the model detect the sampling rate rather than the fault. We therefore use the 48 kHz drive-end fault files, and no resampling is needed.

### Known file quirks

- `99.mat` also contains `X098_DE_time` / `X098_FE_time`. The loader reads the exact key `X099_DE_time`.
- `98.mat` has no RPM variable.
- Downloads from the CWRU server are sometimes truncated. `download.py` validates each file by loading its signal and re-downloads if needed.

## 3. Preprocessing

| Step | Setting |
|---|---|
| Signal | DE accelerometer, raw amplitude, float32 |
| Window | 1024 samples (≈ 21.3 ms at 48 kHz) |
| Stride — 0 HP normal train | 256 (75 % overlap) |
| Stride — normal val / calib / test | 256 |
| Stride — fault recordings | 1024 (no overlap) |
| Normalisation (main pipeline) | z-score with mean/std of the **0 HP normal train segment only**: mean = 0.01255, std = 0.07242 |
| Normalisation (ablation, Setup3) | z-score with mean/std of each load's **calibration segment** (0 HP uses the train segment) |

Normalisation is applied at load time (`src/data/dataset.py`), and the `.npz` files store raw amplitudes. Statistics are in `data/processed/stats.json`.

## 4. Split protocol

Splits are **contiguous in time and never shuffled**. Windows are cut only inside each segment, so overlapping windows cannot leak between splits (verified by `scripts/check_data.py`).

| Recording | Split |
|---|---|
| Normal 0 HP (file 97) | first 70 % → `train`, next 15 % → `val`, last 15 % → `test` |
| Normal 1 / 2 / 3 HP (98, 99, 100) | first 10 % → `calib`, remaining 90 % → `test` |
| Every fault recording | all windows → `test` of its own load |

- The models are trained only on `load0_train` (normal, 0 HP).
- Thresholds are fitted on `load0_val`.
- `calib` is used only for the condition-aware threshold and the normalisation ablation. It is never used for training.

### Resulting window counts

| Load | Split | Normal | Inner race | Ball | Outer race | Total |
|---|---|---|---|---|---|---|
| 0 HP | train | 664 | – | – | – | 664 |
| 0 HP | val | 139 | – | – | – | 139 |
| 0 HP | test | 139 | 238 | 239 | 237 | 853 |
| 1 HP | calib | 186 | – | – | – | 186 |
| 1 HP | test | 1698 | 474 | 475 | 475 | 3122 |
| 2 HP | calib | 186 | – | – | – | 186 |
| 2 HP | test | 1702 | 474 | 475 | 475 | 3126 |
| 3 HP | calib | 186 | – | – | – | 186 |
| 3 HP | test | 1704 | 474 | 477 | 476 | 3131 |

The 0 HP normal recording is only 5 s long (half of the 1–3 HP recordings). The 0 HP test set is therefore fault-heavy (139 normal vs 714 fault windows), which inflates AUC-PR. Report AUC-ROC and the false-positive rate on normal windows alongside AUC-PR, and compare AUC-PR across loads with this prevalence difference in mind.

## 5. Output format

`data/processed/load{L}_{split}.npz` (L ∈ {0,1,2,3}; split ∈ {train, val, test} for 0 HP, {calib, test} for 1–3 HP):

| Array | Shape / type | Meaning |
|---|---|---|
| `X` | (N, 1024) float32 | raw windows |
| `y` | (N,) int8 | 0 = normal, 1 = fault |
| `fault_type` | (N,) str | normal / inner_race / ball / outer_race |
| `fault_size` | (N,) float32 | fault diameter in inches (0 for normal) |
| `file_id` | (N,) int32 | source CWRU file |
| `start` | (N,) int64 | start sample of the window in the source file |

Also produced: `stats.json` (normalisation statistics) and `summary.csv` (window counts).

## 6. SHA-256 checksums (download of 2026-10-06 UTC)

```
16bf48babcf1c7ac224bc1a81cd9eafdb27e42d5cf559761907e067e8eeadf3c  97.mat
37e6612c05e65c415dcfa2ab27a3fda648a5863160fa898b884a14743044e045  98.mat
4b97e6b5361f45efb6951dc3b1aebcdb3b89cb69d0f96d6f5c297dd9f45eee75  99.mat
88a5990cb541320e91505a1d72139e1993500ffe6e292a451011667f4138ca78  100.mat
daddef5f784879becdf8fefbe2453f11c3cfc8d061a2a671da2f546d1bb48460  109.mat
9e2bc579af6f4e6d9d26bf63b7ed84ecba7775740d56a3737cc8f04f73c22d1a  110.mat
a2e64a397a990efb91ba55252a84796f755bda3eb097c4b47cc0730a08dab309  111.mat
1775db11999f268c6ccf04b3ee19857f90a3636f4a582f1abd9a627f991e72d4  112.mat
f392129bf3b43360b964d4e5b3479d0797d1647af8359934b0f56b9abab4739a  122.mat
a1932ec4bad2460238088ca523fdcec6498e562e91f86255863d3987d9fcd558  123.mat
88719243f1b7dbe795c108acaf67ca0115c51517ac35a3fe3460eb07917b2a9a  124.mat
2607d0c291ed9a6138aed3eed7015e133f1cf866ac5cdf2251921f08d058f362  125.mat
5a1c3ceec1f2b58af9e01051e425aa9d31dd018f58a8d31cda25bbdb1ac6dca9  135.mat
37f259eebbc92afa8bd1b445d02af14e83608a48cd7a3cf5c323515ddc2d0f24  136.mat
a81b621153e426aa32848be8157c42bcd81065a28ac7655a9a6597f3889fa3ef  137.mat
b43eeb67badeb129047ddd3658bf673815dc4bd445facb4217eb04653bb2e11d  138.mat
```

If the CWRU server later serves different bytes, `python -m src.data.download --verify-only` followed by a comparison with this list will show it.

## 7. Reproduce the data

```bash
pip install -r requirements.txt
python -m src.data.download      # ~112 MB from the official CWRU server, validated + checksummed
python -m src.data.prepare       # windows, splits, stats -> data/processed/ (~16 MB)
python scripts/check_data.py     # shape / label / no-overlap checks
python -m src.data.data_stats    # per-load statistics + figures (optional)
```

All parameters are in `configs/default.yaml` (`data:` section). The pipeline is deterministic: no random sampling is involved.

## 8. Observed operating-condition shift (normal data)

From `results/tables/data_stats_by_load.csv` and `results/figures/data_psd_by_load.png`:

- **Amplitude.** Normal RMS at 1–3 HP is 10–13 % lower than at 0 HP. A detector fitted on 0 HP amplitudes sees a shifted distribution.
- **Spectrum.** Normal RMS shifts only moderately, but the spectral shape changes with load. The 0 HP spectrum is dominated by a band around 4.1 kHz, whose relative weight drops at higher loads. The band around 5.2 kHz grows with load. The dominant frequency of the averaged spectrum moves from 4.1 kHz (0 HP) to 0.4 kHz (1 HP) and 8.4 kHz (2–3 HP).
- **Faults.** Inner and outer race faults raise RMS by roughly 4–17× and kurtosis to 6–7. The **0.007" ball fault** is the subtle one: RMS only about 2× normal and kurtosis about 3, the same as normal. Expect it to be the main source of missed detections.

## 9. Limitations

- One test rig and one fault size (0.007"). Conclusions may not transfer to other machines or to larger faults.
- The 0 HP normal recording is short (5 s), so the training set is small (664 overlapping windows).
- The operating-condition shift is limited to motor load (0–3 HP, speed 1797→1721 rpm). It does not cover other changes such as temperature or sensor placement.
