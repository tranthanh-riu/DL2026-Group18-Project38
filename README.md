# DL2026-Group18-Project38

## 1. Đề tài và câu hỏi nghiên cứu
Đề #38 — Deep Time-Series Anomaly Detection trên CWRU Bearing Dataset (cảm biến Drive End, 12 kHz). Hai model học chỉ từ dữ liệu normal 0 HP: CNN Autoencoder (score = lỗi dựng lại MSE) và Isolation Forest (8 đặc trưng thống kê).
- **RQ1:** CNN-AE hay Isolation Forest phát hiện lỗi tốt hơn ở 0 HP (Setup1)?
- **RQ2:** Khi đổi tải (1-3 HP) mà giữ nguyên ngưỡng học ở 0 HP, tỉ lệ báo nhầm (FPR) tăng bao nhiêu (Setup2)?
- **RQ3:** Nhiễu (SNR 20/10/5 dB), chiến lược chọn ngưỡng (p99, mean+3σ, POT, per-condition) và chuẩn hóa theo từng tải ảnh hưởng thế nào (Setup3)?

## 2. Cài đặt
Python 3.11 hoặc 3.12.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 3. Dữ liệu
Xem [DATA.md](DATA.md). Cần 16 file `.mat` đặt trong `data/raw/` (tên `<số>.mat`), số file: 97, 98, 99, 100, 105-108, 118-121, 130-133. Tải từ:
```bash
for n in 97 98 99 100 105 106 107 108 118 119 120 121 130 131 132 133; do
  curl -L -o data/raw/$n.mat https://engineering.case.edu/sites/default/files/$n.mat
done
```
(tạo thư mục `data/raw` trước; nếu link không tải được, tải tay từ trang nguồn trong DATA.md).

## 4. Chạy toàn bộ
```bash
bash scripts/run_all.sh
```
Khoảng 1 phút trên CPU (đo được 45 giây, trong đó train 3 seed mất khoảng 4 giây mỗi seed).

## 5. Chạy từng Setup
Chạy từ thư mục gốc repo, theo thứ tự:
```bash
python -m src.data.prepare && python -m src.models.iforest && python -m src.train && python -m src.evaluate   # chuẩn bị + model
python -m src.make_tables    # Setup1 -> results/tables/setup1.csv ; Setup2 -> results/tables/setup2.csv
python -m src.setup3         # Setup3 -> setup3_noise.csv, setup3_threshold.csv, setup3_ablation.csv
python -m src.plots          # hình trong results/figures/
```
Mọi tham số (window, stride, seed, lr, SNR, POT...) nằm trong `configs/default.yaml`.

## 6. Kết quả chính (mean ± std qua 3 seed, ngưỡng = percentile 99 của val 0 HP)
**Setup1 (0 HP)**

| model | load (HP) | auc_roc | auc_pr | precision | recall | f1 | fpr |
|---|---|---|---|---|---|---|---|
| cnn_ae | 0 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| iforest | 0 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.998 ± 0.002 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.019 ± 0.016 |

**Setup2 (1-3 HP)**

| model | load (HP) | auc_roc | auc_pr | precision | recall | f1 | fpr |
|---|---|---|---|---|---|---|---|
| cnn_ae | 1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.990 ± 0.003 | 1.000 ± 0.000 | 0.995 ± 0.002 | 0.009 ± 0.003 |
| cnn_ae | 2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.998 ± 0.003 | 1.000 ± 0.000 | 0.999 ± 0.002 | 0.002 ± 0.003 |
| cnn_ae | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.971 ± 0.008 | 1.000 ± 0.000 | 0.985 ± 0.004 | 0.025 ± 0.007 |
| iforest | 1 | 0.989 ± 0.004 | 0.986 ± 0.005 | 0.796 ± 0.072 | 1.000 ± 0.000 | 0.885 ± 0.044 | 0.220 ± 0.091 |
| iforest | 2 | 0.989 ± 0.005 | 0.986 ± 0.007 | 0.746 ± 0.113 | 1.000 ± 0.000 | 0.852 ± 0.071 | 0.299 ± 0.157 |
| iforest | 3 | 0.993 ± 0.002 | 0.990 ± 0.003 | 0.839 ± 0.072 | 1.000 ± 0.000 | 0.912 ± 0.042 | 0.165 ± 0.082 |

Lưu ý: ở 0 HP cả hai model đều AUC = 1.000 vì lỗi 0.007" có biên độ lớn hơn nhiều so với normal; normal test ở 0 HP chỉ có 35 window nên FPR ở đó thô (bước ~2,9%). Chi tiết Setup3 xem các file `results/tables/setup3_*.csv`.

## 7. Cấu trúc thư mục
```
configs/default.yaml     tham số dùng chung
data/raw/                16 file .mat (không push)
data/processed/          .npz và norm_global.json (không push)
src/data/                inspect_raw.py, prepare.py
src/models/              cnn_ae.py, iforest.py
src/                     features, metrics, train, evaluate, make_tables, noise, thresholds, setup3, plots, utils
checkpoints/             .pt / .joblib (không push)
results/scores|logs|tables|figures
scripts/run_all.sh
DATA.md                  mô tả dữ liệu và split
docs/                    HUONG_DAN.md, TODO.md (kèm ghi chú quyết định)
```
