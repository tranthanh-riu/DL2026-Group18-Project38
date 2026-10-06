# CWRU Dataset

## 1. Official source

Dataset: Case Western Reserve University (CWRU) Bearing Data Center

Official website:
https://engineering.case.edu/bearingdatacenter/welcome

Data used in this project:
- Normal baseline data
- 0.007" inner race fault
- 0.007" ball fault
- 0.007" outer race fault at 6:00 position
- Drive-end (DE) acceleration signals
- Loads: 0 HP, 1 HP, 2 HP, 3 HP

## 2. Download date

Downloaded: 2026-10-06

## 3. Files used

### Normal data

| File | Load |
|---|---:|
| 97.mat | 0 HP |
| 98.mat | 1 HP |
| 99.mat | 2 HP |
| 100.mat | 3 HP |

### 0.007" Inner Race Fault

| File | Load |
|---|---:|
| 105.mat | 0 HP |
| 106.mat | 1 HP |
| 107.mat | 2 HP |
| 108.mat | 3 HP |

### 0.007" Ball Fault

| File | Load |
|---|---:|
| 118.mat | 0 HP |
| 119.mat | 1 HP |
| 120.mat | 2 HP |
| 121.mat | 3 HP |

### 0.007" Outer Race Fault @ 6:00

| File | Load |
|---|---:|
| 130.mat | 0 HP |
| 131.mat | 1 HP |
| 132.mat | 2 HP |
| 133.mat | 3 HP |

Total: 16 MATLAB `.mat` files.

## 4. Sampling rate

The selected Drive-end data use a sampling rate of 12,000 samples/second (12 kHz).

Normal and fault data were selected from the same 12 kHz Drive-end data group, so no resampling was required.

The preprocessing reads the Drive-end signal from the corresponding:

`X***_DE_time`

variable in each `.mat` file.

## 5. Preprocessing

Each Drive-end signal is divided into windows of:

- Window size: 1024 samples

### Normal 0 HP

The 0 HP normal data are split chronologically:

- Train: 70%
- Validation: 15%
- Test: 15%

Window stride:

- Train: 512 samples
- Validation: 1024 samples
- Test: 1024 samples

Validation and test windows are non-overlapping.

### Normal 1/2/3 HP

For each load:

- First 10%: calibration
- Remaining 90%: test

Window stride: 1024 samples.

### Fault data

Fault windows are placed in the test set corresponding to their load.

Fault types:

- Inner race
- Ball
- Outer race @ 6:00

Fault size:

- 0.007"

## 6. Normalization statistics

Mean and standard deviation are calculated only from the training windows of normal 0 HP data.

Saved to:

`data/processed/train_stats.npz`

The main pipeline does not use separate normalization statistics for each load.

## 7. Processed data

Processed files are exported to:

`data/processed/`

with the following naming convention:

`load{L}_{split}.npz`

Each `.npz` file contains:

- `X`
- `y`
- `fault_type`
- `fault_size`

Generated splits include:

- `load0_train.npz`
- `load0_val.npz`
- `load0_test.npz`
- `load1_calibration.npz`
- `load1_test.npz`
- `load2_calibration.npz`
- `load2_test.npz`
- `load3_calibration.npz`
- `load3_test.npz`

## 8. Reproduction

From the project root:

Run:
python src\data\prepare.py

The script reads the raw CWRU `.mat` files and generates the processed `.npz` files under:

data/processed/