# Hướng dẫn từng bước – Đề #38 Time-Series Anomaly Detection (Group 18)

Repo: DL2026-Group18-Project38. Tài liệu này là đặc tả để viết code; Claude Code đọc cùng với CLAUDE.md.

## Cách dùng tài liệu

Làm tuần tự từ Bước 0 đến Bước 8; không sang bước sau khi chưa qua mục **Tự kiểm tra** của bước trước. Mỗi bước gồm: file cần viết, các hàm bên trong, gợi ý cách làm, và điểm kiểm tra.

Scope giữ lại: Setup1, Setup2, Setup3 (noise, 4 ngưỡng, ablation), 3 seed.
Scope bỏ: phân tích FFT theo tải, feature drift, quét độ nhạy ngưỡng, demo inference.

Dữ liệu đã được tải thủ công vào `data/raw/` (16 file: 97, 98, 99, 100, 105-108, 118-121, 130-133, tên `<số>.mat`). **Không viết `download.py`.**

Thứ tự và thời lượng (tổng khoảng 11 giờ code):

1. Bước 0 — chuẩn bị repo: 20 phút
2. Bước 1 — dữ liệu: 2 giờ
3. Bước 2 — Isolation Forest + metric: 1,5 giờ
4. Bước 3 — CNN-AE: 2 giờ (cho train chạy nền, viết tiếp Bước 4-5 trong lúc chờ)
5. Bước 4 — bảng Setup1, Setup2: 1 giờ
6. Bước 5 — Setup3: 2,5 giờ
7. Bước 6 — hình: 1 giờ
8. Bước 7-8 — run_all, README, chạy thử repo sạch, nộp: 1 giờ

Commit và push sau mỗi bước qua được điểm kiểm tra.

---

## Bước 0 — Chuẩn bị repo (20 phút)

Mục tiêu: repo cài được thư viện, có config chung và hàm đặt seed mà mọi script sau đều dùng.

1. **Nhánh riêng.** Clone repo, tạo nhánh từ `main`:

```bash
git clone https://github.com/tranthanh-riu/DL2026-Group18-Project38.git
cd DL2026-Group18-Project38
git checkout -b mieu-code
```

2. **Môi trường ảo (Python 3.11 hoặc 3.12).** Tạo `.venv`, kích hoạt, rồi cài torch bản CPU trước (nhẹ hơn nhiều so với bản CUDA):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

3. **`requirements.txt`**: numpy, scipy, scikit-learn, pandas, matplotlib, pyyaml, torch.
4. **`.gitignore`**: `.venv/`, `data/raw/`, `data/processed/`, `checkpoints/*.pt`, `results/scores/`, `__pycache__/`. Không bao giờ push file `.mat`, `.npz`, `.pt`.
5. **`configs/default.yaml`** — mọi con số đọc từ đây, không gõ cứng trong code:

```yaml
window: 1024
train_stride: 512
eval_stride: 1024
split: [0.70, 0.15, 0.15]
calib_frac: 0.10
seeds: [0, 1, 2]
snr_db: [20, 10, 5]
ae: {lr: 0.001, batch: 64, epochs: 50, patience: 5}
iforest: {n_estimators: 200}
pot: {init_quantile: 0.98, q: 0.001}
```

6. **`src/utils.py`** — viết hai hàm:
    - `load_config(path="configs/default.yaml")`: mở file, `yaml.safe_load`, trả về dict.
    - `set_seed(seed)`: gọi `random.seed`, `np.random.seed`, `torch.manual_seed` với cùng giá trị.

**Tự kiểm tra:**
`python -c "import torch, sklearn, scipy; print(torch.__version__)"` chạy không lỗi, và
`python -c "from src.utils import load_config; print(load_config()['seeds'])"` in ra `[0, 1, 2]`.
Để import kiểu `src.utils` chạy được, thêm file rỗng `src/__init__.py` và luôn chạy lệnh từ thư mục gốc repo (`python -m src.data.prepare`).

---

## Bước 1 — Dữ liệu CWRU (2 giờ)

Mục tiêu: từ 16 file `.mat` ra các file `.npz` đã chia split và chuẩn hóa, cùng bảng số window cho report.

### 1a. Dữ liệu thô (đã tải tay — bỏ qua download.py)

Nguồn chính thức (ghi vào DATA.md):
- https://engineering.case.edu/bearingdatacenter/normal-baseline-data
- https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data

Mẫu URL: `https://engineering.case.edu/sites/default/files/<số>.mat`. Ngày tải: 06/10/2026.

Ánh xạ số file (khớp bảng trên hai trang nguồn):

| Loại | 0 HP | 1 HP | 2 HP | 3 HP |
| --- | --- | --- | --- | --- |
| Normal | 97 | 98 | 99 | 100 |
| Inner race 0,007" (IR007_*) | 105 | 106 | 107 | 108 |
| Ball 0,007" (B007_*) | 118 | 119 | 120 | 121 |
| Outer race @6:00 0,007" (OR007@6_*) | 130 | 131 | 132 | 133 |

Trong code dùng dict:

```python
FILES = {
    0: {"normal": 97,  "IR": 105, "B": 118, "OR": 130},
    1: {"normal": 98,  "IR": 106, "B": 119, "OR": 131},
    2: {"normal": 99,  "IR": 107, "B": 120, "OR": 132},
    3: {"normal": 100, "IR": 108, "B": 121, "OR": 133},
}
```

### 1b. `src/data/inspect_raw.py` — kiểm tra dữ liệu thô (làm TRƯỚC prepare.py)

Với mỗi file trong `data/raw/`, in: số file, tên biến `_DE_time`, độ dài tín hiệu, thời lượng nếu 12 kHz (`len / 12000`) và nếu 48 kHz (`len / 48000`).

Kiểm tra:
- Tên biến phải khớp số file (ví dụ `105.mat` chứa `X105_DE_time`).
- Mỗi file chỉ có đúng một biến `_DE_time`.
- Trang CWRU nói dữ liệu DE được thu ở cả 12 kHz và 48 kHz; trang không ghi tần số của nhóm Normal Baseline. Normal và fault **phải cùng sampling rate**.
- Nếu lệch: hạ tần số file cao hơn xuống bằng `scipy.signal.resample_poly` và ghi lý do vào DATA.md. **Nếu có điều gì bất thường, DỪNG và hỏi người dùng.**

### 1c. `src/data/prepare.py` — đọc tín hiệu

Viết `load_de(n)`:

1. `m = loadmat(f"data/raw/{n}.mat")`.
2. Lọc các key kết thúc bằng `_DE_time`. Ưu tiên key `f"X{n:03d}_DE_time"`; nếu không có thì lấy key đầu tiên và `print` cảnh báo.
3. Trả về `m[key].ravel().astype(np.float32)`.

### 1d. Cắt window

Viết `make_windows(sig, size, stride)` trả về mảng `(N, size)`. Một dòng là đủ:

```python
np.lib.stride_tricks.sliding_window_view(sig, size)[::stride]
```

Thêm `.copy()` ở cuối để mảng không còn là view của tín hiệu gốc.

### 1e. Chia split theo thời gian

Nguyên tắc: **cắt tín hiệu trước, cắt window sau**, để không window nào nằm vắt qua hai split.

| Dữ liệu | Cách chia | Stride | Nhãn |
| --- | --- | --- | --- |
| Normal 0 HP | 70% đầu → train, 15% tiếp → val, 15% cuối → test | train 512; val/test 1024 | 0 |
| Normal 1/2/3 HP | 10% đầu → calib, 90% còn lại → test | 1024 | 0 |
| Fault (IR, B, OR) mỗi tải | toàn bộ → test của tải đó | 1024 | 1 |

Với tín hiệu dài `n`: train là `sig[:int(0.70*n)]`, val là `sig[int(0.70*n):int(0.85*n)]`, test là `sig[int(0.85*n):]`.

In số window của val và calib. **Nếu chỉ vài chục window**, ngưỡng percentile 99 và POT ở Bước 5 sẽ không ổn định. Cách xử lý: tạo thêm bản val/calib với stride 256 chỉ để ước lượng ngưỡng. Không rò rỉ dữ liệu vì các window này vẫn nằm trọn trong đoạn val/calib. Ghi quyết định này vào DATA.md.

### 1f. Chuẩn hóa

1. Tính `mean`, `std` là **hai số vô hướng** từ toàn bộ giá trị của train 0 HP (tín hiệu thô).
2. Lưu vào `data/processed/norm_global.json`.
3. Áp dụng `X = (X_raw - mean) / std` cho mọi split của mọi tải.

### 1g. Xuất file

| File | Nội dung |
| --- | --- |
| `load0_train.npz`, `load0_val.npz` | chỉ normal 0 HP |
| `load{L}_test.npz`, L = 0..3 | normal test + 3 loại fault của tải L |
| `load{L}_calib.npz`, L = 1..3 | 10% đầu normal của tải L |

Mỗi file có các khóa: `X` (N×1024, float32, đã chuẩn hóa), `X_raw` (chưa chuẩn hóa — Bước 5 cần để thêm nhiễu và làm ablation), `y` (0/1), `fault_type` (`"normal"`, `"IR"`, `"B"`, `"OR"`).

Cuối script, gom số window vào một DataFrame và in bảng theo tải × split × fault_type (`groupby(...).size()`). Bảng này đưa vào DATA.md và mục Dataset của report.

**Tự kiểm tra:**

- [ ] `load0_train.npz` có `X.shape == (N, 1024)`, không có NaN (`np.isnan(X).any()` là `False`).
- [ ] Sau chuẩn hóa, `X` của train 0 HP có mean ≈ 0 và std ≈ 1.
- [ ] Mỗi `load{L}_test.npz` có cả `y == 0` lẫn `y == 1`.
- [ ] DATA.md có URL, ngày tải, danh sách 16 file, sampling rate, bảng split, lệnh tái tạo (`python -m src.data.prepare`).

---

## Bước 2 — Baseline Isolation Forest và metric (1,5 giờ)

Mục tiêu: score của Isolation Forest cho val, calib và test 4 tải, 3 seed; cùng module metric dùng chung cho cả đề.

### 2a. `src/features.py`

Viết `extract_features(X)` nhận `(N, 1024)`, trả về `(N, 8)`. Mọi phép tính dùng `axis=1` (theo từng window):

| Đặc trưng | Công thức |
| --- | --- |
| RMS | căn bậc hai của trung bình x² |
| std | `X.std(axis=1)` |
| mean abs | trung bình giá trị tuyệt đối |
| peak | max giá trị tuyệt đối |
| peak-to-peak | max − min |
| crest factor | peak / RMS |
| skewness | `scipy.stats.skew(X, axis=1)` |
| kurtosis | `scipy.stats.kurtosis(X, axis=1)` |

Ghép bằng `np.stack([...], axis=1)`. Dùng `X` đã chuẩn hóa để nhất quán với CNN-AE.

### 2b. `src/models/iforest.py`

1. `fit(F_train, seed)`: tạo `IsolationForest(n_estimators=200, random_state=seed)`, `.fit(F_train)`, trả model.
2. `score(model, F)`: trả `-model.score_samples(F)`. Đổi dấu để **score càng cao càng bất thường**, cùng chiều với MSE của CNN-AE.
3. Hàm `main()`: với mỗi seed, fit trên feature của `load0_train`, rồi tính và lưu score:
    - `results/scores/iforest_seed{s}_val.npz` — khóa `score`
    - `results/scores/iforest_seed{s}_calib{L}.npz`, L = 1..3 — khóa `score`
    - `results/scores/iforest_seed{s}_load{L}.npz`, L = 0..3 — khóa `score`, `y`, `fault_type`

Dùng đúng cách đặt tên này cho CNN-AE (thay `iforest` bằng `cnn_ae`), để Bước 4-5 chỉ cần một vòng lặp cho cả hai model.

### 2c. `src/metrics.py`

Hai hàm, trả về dict:

1. `threshold_free(y, score)` → `auc_roc` (`roc_auc_score`), `auc_pr` (`average_precision_score`).
2. `at_threshold(y, score, thr)`:
    - `pred = (score > thr).astype(int)`
    - `precision`, `recall`, `f1` bằng các hàm sklearn cùng tên, thêm `zero_division=0`.
    - `fpr` = số window có `pred == 1` và `y == 0`, chia cho số window có `y == 0`.

**Tự kiểm tra:**

- [ ] Test nhanh `at_threshold` với dữ liệu tự bịa: `y = [0,0,1,1]`, `score = [0.1,0.9,0.8,0.2]`, `thr = 0.5` → precision 0,5, recall 0,5, FPR 0,5.
- [ ] AUC-ROC của Isolation Forest ở 0 HP **cao rõ rệt hơn 0,5**. Nếu gần hoặc dưới 0,5, kiểm tra lại dấu của score.

---

## Bước 3 — CNN Autoencoder (2 giờ)

Mục tiêu: 3 checkpoint (seed 0, 1, 2) train trên normal 0 HP, và file score theo đúng cách đặt tên của Bước 2.

### 3a. `src/models/cnn_ae.py` — kiến trúc

Mọi lớp dùng `kernel_size=4, stride=2, padding=1`. Encoder có ReLU sau mỗi lớp; decoder có ReLU sau hai lớp đầu, lớp cuối **không** có activation (tín hiệu chuẩn hóa có giá trị âm).

| Lớp | Loại | Kênh vào → ra | Độ dài |
| --- | --- | --- | --- |
| enc1 | Conv1d | 1 → 16 | 1024 → 512 |
| enc2 | Conv1d | 16 → 32 | 512 → 256 |
| enc3 | Conv1d | 32 → 64 | 256 → 128 |
| dec1 | ConvTranspose1d | 64 → 32 | 128 → 256 |
| dec2 | ConvTranspose1d | 32 → 16 | 256 → 512 |
| dec3 | ConvTranspose1d | 16 → 1 | 512 → 1024 |

Cột độ dài tự kiểm bằng hai công thức. Công thức của Conv1d là công thức trong slide Bài 6, viết cho một chiều:

```latex
L_{out} = \frac{L_{in} - F + 2P}{S} + 1
```

ConvTranspose1d làm phép ngược, nên độ dài tăng:

```latex
L_{out} = (L_{in} - 1)\,S - 2P + F
```

Ví dụ dec1: (128 − 1)·2 − 2 + 4 = 256.

Khung class:

```python
class CNNAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            # enc2, enc3 ...
        )
        self.decoder = nn.Sequential(
            # dec1, dec2 (có ReLU), dec3 (không ReLU)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
```

**Tự kiểm tra:** `CNNAE()(torch.randn(8, 1, 1024)).shape` phải là `torch.Size([8, 1, 1024])`. Đếm tham số bằng `sum(p.numel() for p in model.parameters())` và ghi vào report.

### 3b. `src/train.py`

Trình tự trong `train_one_seed(seed)`:

1. `set_seed(seed)`.
2. Đọc `X` của `load0_train.npz` và `load0_val.npz`. Đổi sang tensor rồi thêm chiều kênh: `torch.from_numpy(X).unsqueeze(1)` → `(N, 1, 1024)`.
3. `DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)`; val dùng `shuffle=False`.
4. Tạo model, `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`, `criterion = nn.MSELoss()`.
5. Mỗi epoch:
    1. `model.train()`. Với mỗi batch: `optimizer.zero_grad()` → `loss = criterion(model(xb), xb)` → `loss.backward()` → `optimizer.step()`.
    2. `model.eval()` và `with torch.no_grad():` tính val loss trung bình.
    3. Val loss tốt hơn tốt nhất cũ thì lưu `torch.save(model.state_dict(), f"checkpoints/cnn_ae_seed{seed}.pt")` và đặt lại bộ đếm; không thì cộng 1. Bộ đếm chạm 5 thì dừng (early stopping).
6. Ghi epoch, train loss, val loss vào `results/logs/train_seed{seed}.csv`; ghi tổng thời gian train.

Nối với slide: target của loss chính là input (`criterion(model(xb), xb)`) — đây là điểm khác duy nhất so với bài toán có nhãn. `loss.backward()` là backpropagation của Bài 5; `optimizer.step()` là bước cập nhật trọng số của gradient descent Bài 2.

### 3c. `src/evaluate.py`

1. `reconstruction_score(model, X)`: chạy theo batch trong `torch.no_grad()`, mỗi window tính `((x - x_hat) ** 2).mean(dim=(1, 2))`, nối lại thành mảng numpy dài N.
2. Với mỗi seed: load checkpoint (`model.load_state_dict(torch.load(path))`, rồi `model.eval()`), tính và lưu score cho val, calib 1-3, test 0-3 với tên `cnn_ae_seed{s}_...npz` giống hệt Bước 2.

**Tự kiểm tra:**

- [ ] Val loss giảm rồi phẳng lại; không phải NaN.
- [ ] Ở 0 HP, score trung bình của window fault cao hơn rõ rệt so với window normal.
- [ ] Có đủ 3 checkpoint và đủ file score cho cả 3 seed.

---

## Bước 4 — Bảng Setup1 và Setup2 (1 giờ)

Mục tiêu: `results/tables/setup1.csv` và `setup2.csv` dạng mean ± std qua 3 seed. Bước này không train gì, chỉ đọc file score.

### `src/make_tables.py`

1. Ba vòng lặp lồng nhau: `model` trong `["iforest", "cnn_ae"]`, `seed` trong `[0, 1, 2]`, `L` trong `[0, 1, 2, 3]`.
2. Trong vòng lặp:
    1. Đọc val score, tính `thr = np.percentile(val_score, 99)`.
    2. Đọc test score của tải L.
    3. Gọi `threshold_free` và `at_threshold`, gộp hai dict, thêm `model`, `seed`, `load`, rồi `append` vào list `rows`.
3. `df = pd.DataFrame(rows)`, sau đó `df.groupby(["model", "load"]).agg(["mean", "std"])`.
4. Viết hàm nhỏ định dạng mỗi ô thành chuỗi `"0.912 ± 0.004"`.
5. Hàng `load == 0` → `setup1.csv`; hàng `load` 1-3 → `setup2.csv`.

Viết luôn hàm `summarize(df, group_cols)` dùng chung — Bước 5 gọi lại hàm này cho 3 bảng Setup3.

### Đọc kết quả

- **RQ1:** so AUC-ROC và AUC-PR của hai model ở 0 HP. Model nào cao hơn, chênh bao nhiêu, std có chồng nhau không.
- **RQ2:** FPR của mỗi model đi từ 0 HP sang 1, 2, 3 HP. Câu cần viết trong Abstract có dạng: "FPR của CNN-AE tăng từ x% ở 0 HP lên y% ở 3 HP."

FPR tăng mạnh ở tải khác **không phải là bug** — đó chính là kết quả đề muốn bạn đo.

**Tự kiểm tra:**

- [ ] Mỗi ô có dạng mean ± std; std không bằng 0 ở mọi ô (nếu bằng 0, kiểm tra seed có thật sự thay đổi không).
- [ ] Số liệu 0 HP trong `setup2.csv` (nếu có) khớp với `setup1.csv`.

---

## Bước 5 — Setup3: noise, 4 ngưỡng, ablation (2,5 giờ)

Mục tiêu: `setup3_noise.csv`, `setup3_threshold.csv`, `setup3_ablation.csv`. Không train lại model nào; mọi thứ chạy ở bước inference.

Để code gọn, viết trước một hàm `score_any(model_name, seed, X)` trả score cho cả hai model: với `iforest` thì gọi `extract_features` rồi `score`; với `cnn_ae` thì gọi `reconstruction_score`. Hàm này cũng cần lưu lại model Isolation Forest đã fit ở Bước 2 (dùng `joblib.dump`), hoặc fit lại với cùng seed.

### 5a. Noise — `src/noise.py`

Viết `add_noise(X_raw, snr_db, seed)`:

1. `P = (X_raw ** 2).mean(axis=1, keepdims=True)` — công suất từng window.
2. Độ lệch chuẩn nhiễu theo SNR:

```latex
\sigma = \sqrt{\frac{P}{10^{SNR/10}}}
```

3. `rng = np.random.default_rng(seed)`, trả về `X_raw + rng.standard_normal(X_raw.shape) * sigma`.

Quy trình: với mỗi SNR (20, 10, 5), seed, tải, model → thêm nhiễu vào `X_raw` của test → chuẩn hóa bằng `norm_global.json` → `score_any` → ngưỡng percentile 99 của val **sạch** → metric AUC-PR, F1, FPR. Nhiễu chỉ vào test; val và train giữ nguyên.

### 5b. Bốn chiến lược ngưỡng — `src/thresholds.py`

Mỗi hàm nhận mảng score, trả về một số:

1. `thr_p99(val)` → `np.percentile(val, 99)`.
2. `thr_mean3s(val)` → `val.mean() + 3 * val.std()`.
3. `thr_pot(val, init_q=0.98, q=1e-3)` — chi tiết bên dưới.
4. `thr_per_condition(calib_L)` → percentile 99 của score calib của **chính tải đó**. Với 0 HP dùng val.

Thêm `thr_oracle(y, score)` chỉ để tham khảo: dùng `precision_recall_curve`, tính F1 tại mọi ngưỡng, lấy ngưỡng có F1 cao nhất. Report phải ghi rõ đây là cận trên, vì nó nhìn nhãn test.

**POT từng bước:**

1. `t = np.quantile(val, init_q)` — ngưỡng khởi đầu.
2. `excess = val[val > t] - t`; `N_t = len(excess)`, `n = len(val)`.
3. `xi, _, sigma = scipy.stats.genpareto.fit(excess, floc=0)` — hàm trả về (shape ξ, loc, scale σ).
4. Ngưỡng cuối cùng:

```latex
z_q = t + \frac{\sigma}{\xi}\left[\left(\frac{q\,n}{N_t}\right)^{-\xi} - 1\right]
```

5. Nếu |ξ| rất nhỏ (dưới 1e-6), dùng dạng giới hạn: `t - sigma * np.log(q * n / N_t)`.
6. In ra `N_t`, ξ, σ, `z_q`. Nếu `N_t` dưới khoảng 10, kết quả không đáng tin: hạ `init_q` xuống 0,95 hoặc 0,90, hoặc dùng bản val stride 256 (Bước 1e). Ghi lựa chọn vào report.

Quy trình: với mỗi chiến lược, seed, tải → metric Precision, Recall, F1, FPR, chủ yếu cho CNN-AE (thêm IF nếu kịp) → `setup3_threshold.csv`.

### 5c. Ablation chuẩn hóa

Với tải L = 1, 2, 3 và CNN-AE:

1. Tính `mean_L`, `std_L` từ `X_raw` của `load{L}_calib.npz`.
2. Chuẩn hóa lại `X_raw` của test và calib tải L bằng `mean_L`, `std_L`.
3. Tính score, đặt ngưỡng percentile 99 trên score calib (cùng kiểu chuẩn hóa), lấy AUC-PR, F1, FPR.
4. Ghép với kết quả chuẩn hóa toàn cục (đã có từ Bước 4) thành hai cột so sánh → `setup3_ablation.csv`.

**Tự kiểm tra:**

- [ ] SNR càng thấp, AUC-PR nhìn chung càng giảm. Nếu không, kiểm tra nhiễu có thật sự được cộng vào trước bước chuẩn hóa không.
- [ ] Ngưỡng POT lớn hơn `t` và cùng cỡ với p99; không phải NaN hay số khổng lồ.
- [ ] Oracle F1 ≥ F1 của mọi chiến lược khác ở cùng tải — nếu không, có lỗi trong code.

---

## Bước 6 — Hình và phân tích (1 giờ)

Mục tiêu: 6 hình trong `results/figures/` và 4-6 ca lỗi để viết mục Error Analysis. Mọi hình lưu bằng `plt.savefig(path, dpi=200, bbox_inches="tight")`, rồi `plt.close()`.

### `src/plots.py` — mỗi hình một hàm

| Hình | Cách vẽ | Trả lời câu hỏi |
| --- | --- | --- |
| Đường loss | train và val loss theo epoch, seed 0, đọc từ `results/logs/` | Model có hội tụ không (RQ1, Methods) |
| Phân bố score normal theo tải | 4 histogram chồng nhau (`alpha=0.5`) của score window normal, CNN-AE seed 0, kẻ đường dọc tại ngưỡng p99 | Vì sao đổi tải làm báo nhầm (RQ2) |
| FPR theo tải | trục x là 0-3 HP, một đường cho mỗi model, thanh sai số = std (`plt.errorbar`) | Mức suy giảm khi đổi tải (RQ2) |
| F1 theo SNR | trục x là sạch/20/10/5 dB, một đường cho mỗi model | Độ bền với nhiễu (RQ3) |
| F1 theo chiến lược ngưỡng | cột nhóm: nhóm theo tải, mỗi cột một chiến lược | Ngưỡng nào bền nhất (RQ3) |
| Gốc và tái tạo | mỗi tải một ô con: tín hiệu gốc và `x_hat` của một window normal chồng lên nhau | Giải thích trực quan reconstruction error |

### Chọn ca lỗi

Với CNN-AE seed 0 và ngưỡng p99:

1. **Fault bị bỏ sót:** window có `y == 1` và `score <= thr`. In ra `fault_type` của chúng — loại lỗi nào hay bị bỏ sót?
2. **Normal báo nhầm ở 3 HP:** window có `y == 0` và `score > thr`.
3. **Ca được cứu nhờ chuẩn hóa theo điều kiện:** window báo nhầm khi chuẩn hóa toàn cục nhưng đúng sau khi chuẩn hóa bằng calib (so hai mảng dự đoán từ Bước 5c).

Với mỗi ca, vẽ tín hiệu gốc, tín hiệu tái tạo, và sai số theo thời gian `(x - x_hat) ** 2`. Viết 2-3 câu: lỗi gì, vì sao xảy ra, gợi ý cải thiện.

**Tự kiểm tra:**

- [ ] Mọi hình có tiêu đề trục kèm đơn vị và chú thích (legend) khi có hơn một đường.
- [ ] Số liệu trên hình khớp với CSV trong `results/tables/`.

---

## Bước 7-8 — Đóng gói, kiểm tra, nộp (1 giờ)

Mục tiêu: người lạ clone repo, làm theo README, ra đúng bảng và hình trong report.

### 7a. `scripts/run_all.sh`

Giữ `set -e` ở đầu (dừng ngay khi một bước lỗi), rồi gọi lần lượt (dữ liệu thô đã có trong `data/raw/`, không có bước tải):

```bash
python -m src.data.prepare
python -m src.models.iforest
python -m src.train
python -m src.evaluate
python -m src.setup3        # hoặc tên file bạn đặt cho Bước 5
python -m src.make_tables
python -m src.plots
```

Mỗi file Python cần khối `if __name__ == "__main__": main()` ở cuối để chạy được bằng `python -m`.

### 7b. README.md

Đổi tiêu đề thành `DL2026-Group18-Project38`, rồi có đủ các mục:

1. Đề tài và 3 câu hỏi nghiên cứu (2-3 dòng).
2. Cài đặt: lệnh tạo venv và `pip install`.
3. Dữ liệu: trỏ sang DATA.md; kèm lệnh tải 16 file (mẫu URL `https://engineering.case.edu/sites/default/files/<số>.mat`, danh sách số file ở Bước 1a).
4. Chạy toàn bộ: `bash scripts/run_all.sh`, kèm thời gian chạy ước tính trên CPU.
5. Chạy từng Setup: lệnh riêng cho Setup1, Setup2, Setup3.
6. Bảng kết quả chính (copy từ `setup1.csv` và `setup2.csv`).
7. Cấu trúc thư mục.

### 8a. Chạy thử trên repo sạch

1. Push hết, mở PR vào `main`.
2. Clone vào một thư mục trống khác, tạo venv mới, làm đúng theo README.
3. So số trong `setup1.csv` mới với số trong report. Lệch thì sửa trước khi nộp.

### 8b. Checklist nộp

- [ ] Report PDF 10-15 trang, đặt tên `GroupID_ProjectID_Report.pdf` theo yêu cầu đề.
- [ ] Repo public hoặc đã cấp quyền cho giảng viên.
- [ ] README, DATA.md, `run_all.sh` chạy được.
- [ ] Mọi con số trong report khớp CSV.
- [ ] References 8-10 nguồn, trích dẫn khớp hai chiều.
- [ ] Bảng đóng góp thành viên.
- [ ] Leader nộp report, link repo và link dataset lên Google Classroom trước 01:00 (hạn chính thức 08:00).

---

## Lỗi hay gặp và cách sửa

| Triệu chứng | Nguyên nhân thường gặp | Cách sửa |
| --- | --- | --- |
| `KeyError` khi đọc `.mat` | Tên biến trong file không khớp số file | Lọc key kết thúc bằng `_DE_time` (Bước 1c) |
| `RuntimeError: Expected 3D input` ở Conv1d | Quên thêm chiều kênh | `X.unsqueeze(1)` để có `(N, 1, 1024)` |
| Output decoder dài 1022 hoặc 1026 | Sai `padding` hoặc `kernel_size` | Dùng đúng `k=4, s=2, p=1` cho mọi lớp; kiểm lại bằng công thức Bước 3a |
| Loss là NaN hoặc tăng vọt | Chưa chuẩn hóa, hoặc learning rate quá lớn | Kiểm tra mean ≈ 0, std ≈ 1; thử `lr=1e-4` |
| Val loss không giảm | Quên `optimizer.zero_grad()` hoặc `optimizer.step()` | Đối chiếu vòng train với Bước 3b |
| AUC-ROC dưới 0,5 | Score ngược dấu (thường là Isolation Forest) | Dùng `-score_samples` |
| Hầu hết window normal ở 1-3 HP bị báo bất thường | Đây là condition shift — kết quả cần đo, không phải bug | Ghi lại, giải thích trong Discussion, so với ablation |
| `genpareto.fit` cảnh báo hoặc ngưỡng POT vô lý | Quá ít điểm vượt `t` | Hạ `init_q`, hoặc dùng val stride 256 (Bước 1e, 5b) |
| Lỡ push file `.mat`, `.npz` hoặc `.pt` | Thiếu `.gitignore` | Thêm vào `.gitignore`, rồi `git rm --cached <file>` và commit |
| `ModuleNotFoundError: src` | Chạy script từ sai thư mục | Chạy từ thư mục gốc bằng `python -m src.<module>`; có `src/__init__.py` |
