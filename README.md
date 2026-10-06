# Deep Time-Series Anomaly Detection under Changing Conditions

Unsupervised anomaly detection on CWRU bearing vibration data: models are trained on normal data at 0 HP and evaluated when the motor load changes (1–3 HP), under added noise, and with different detection thresholds.

> Status: **data pipeline done**. Baseline, CNN-AE, robustness experiments, and analysis are still to be added.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.12, numpy, scipy 1.17, pandas, matplotlib, PyYAML. CPU is enough.

## Data

```bash
python -m src.data.download      # download + validate 16 CWRU files (~112 MB)
python -m src.data.prepare       # windows, time-ordered splits, normalisation stats
python scripts/check_data.py     # sanity checks
python -m src.data.data_stats    # per-load statistics and figures
```

Details (source, version/checksums, split, preprocessing): see [DATA.md](DATA.md).

Load data in any later step:

```python
from src.data.dataset import load_split
train = load_split(0, "train")                      # X: (N, 1, 1024), z-scored with 0 HP train stats
test2 = load_split(2, "test")                       # normal + fault windows at 2 HP
test2_c = load_split(2, "test", norm="condition")   # ablation: per-load calibration stats
```

## Repository layout

```
configs/default.yaml     all parameters
data/manifest.csv        the 16 CWRU files used
src/data/download.py     download + integrity check + SHA-256
src/data/prepare.py      windowing, splits, stats -> data/processed/
src/data/dataset.py      load_split() used by all models
src/data/data_stats.py   per-load statistics and figures
scripts/check_data.py    data sanity checks
results/                 tables and figures
```

