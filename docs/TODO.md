# TODO — Đề #38 (làm lần lượt từ trên xuống)

## Cách dùng file này (dành cho Claude Code)

1. Đọc `CLAUDE.md` và `docs/HUONG_DAN.md` trước. File này chỉ là danh sách việc và thứ tự;
   chi tiết từng bước nằm trong `HUONG_DAN.md` (mỗi mục ghi rõ "xem Bước ...").
2. Lấy **mục chưa đánh dấu đầu tiên** (`[ ]`). Làm đúng mục đó, không làm trước mục sau.
3. Làm xong: chạy phần kiểm tra của mục, in kết quả cho người dùng xem,
   đổi `[ ]` thành `[x]` trong file này, rồi commit (một commit nhỏ cho mỗi mục hoặc nhóm mục liền nhau).
4. Gặp dòng **🛑 DỪNG**: tóm tắt kết quả, rồi **dừng hẳn** và chờ người dùng gõ "tiếp".
   Không tự đi tiếp qua dòng 🛑 dù mọi thứ có vẻ ổn.
5. Gặp điều bất thường (tên biến lệch số file, normal/fault khác tần số, ít window,
   kết quả vô lý): dừng và hỏi người dùng, không tự quyết.
6. Sau mỗi file code viết xong, giải thích ngắn 3-5 dòng file làm gì.
7. Mọi quyết định đáng nhớ (resample, stride 256, đổi `init_q` của POT...) ghi vào mục
   "Ghi chú quyết định" ở cuối file này, để đưa vào DATA.md và report.

Dữ liệu đã có sẵn trong `data/raw/` (97-100, 105-108, 118-121, 130-133). Không tải lại, không viết `download.py`.

---

## Giai đoạn 0 — Chuẩn bị repo (HUONG_DAN: Bước 0)

- [x] 0.1 Tạo `requirements.txt`, `.gitignore`, và các file rỗng `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`
- [x] 0.2 Điền `configs/default.yaml` (window, stride, split, calib_frac, seeds, snr_db, ae, iforest, pot)
- [x] 0.3 Viết `src/utils.py`: `load_config`, `set_seed`
- [x] 0.4 Chạy hai lệnh "Tự kiểm tra" của Bước 0 và in kết quả

---

## Giai đoạn 1 — Dữ liệu (HUONG_DAN: Bước 1)

- [x] 1.1 Viết `src/data/inspect_raw.py` (Bước 1b): in số file, tên biến `_DE_time`, độ dài, thời lượng nếu 12 kHz và 48 kHz. Chạy và in toàn bộ 16 dòng.

🛑 **DỪNG 1** — người dùng đối chiếu: tên biến khớp số file? mỗi file đúng một biến `_DE_time`? nhóm Normal (97-100) cùng tần số với nhóm fault? Chờ "tiếp". Nếu lệch tần số: hỏi người dùng chọn cách xử lý (resample), ghi vào "Ghi chú quyết định".

- [x] 1.2 `src/data/prepare.py`: viết `load_de(n)` và `make_windows(sig, size, stride)` (Bước 1c, 1d)
- [x] 1.3 Chia split theo thời gian: cắt tín hiệu trước, cắt window sau (Bước 1e). Train/val/test cho normal 0 HP; calib/test cho normal 1-3 HP; fault vào test.
- [x] 1.4 Chuẩn hóa global từ train 0 HP, lưu `data/processed/norm_global.json` (Bước 1f)
- [x] 1.5 Xuất các file `.npz` với khóa `X`, `X_raw`, `y`, `fault_type` (Bước 1g); in bảng số window theo tải x split x fault_type
- [x] 1.6 Chạy các mục "Tự kiểm tra" của Bước 1 và in kết quả (shape, NaN, mean/std, y có cả 0 và 1)

🛑 **DỪNG 2** — người dùng xem bảng số window. Nếu val hoặc calib chỉ vài chục window: hỏi có dùng stride 256 để ước lượng ngưỡng không (Bước 1e). Chờ "tiếp".

- [x] 1.7 Viết nháp `DATA.md`: URL, ngày tải (06/10/2026), 16 file, sampling rate đã xác nhận, bảng split, lệnh tái tạo

---

## Giai đoạn 2 — Baseline Isolation Forest và metric (HUONG_DAN: Bước 2)

- [ ] 2.1 `src/features.py`: `extract_features(X)` trả `(N, 8)`; test shape
- [ ] 2.2 `src/metrics.py`: `threshold_free`, `at_threshold`; chạy test bịa (precision 0,5, recall 0,5, FPR 0,5)
- [ ] 2.3 `src/models/iforest.py`: `fit`, `score`, `main()`; chạy 3 seed, lưu score đúng tên file ở Bước 2b
- [ ] 2.4 In AUC-ROC và AUC-PR của Isolation Forest ở 0 HP (3 seed)

🛑 **DỪNG 3** — người dùng kiểm tra AUC-ROC cao rõ rệt hơn 0,5. Nếu gần hoặc dưới 0,5: kiểm tra dấu của score trước khi báo. Chờ "tiếp".

---

## Giai đoạn 3 — CNN Autoencoder (HUONG_DAN: Bước 3)

- [ ] 3.1 `src/models/cnn_ae.py`: kiến trúc theo bảng Bước 3a; test `torch.randn(8,1,1024)` ra đúng shape; in số tham số
- [ ] 3.2 `src/train.py`: viết `train_one_seed(seed)` (Adam, MSE, early stopping patience 5, lưu checkpoint tốt nhất, ghi CSV loss và thời gian)
- [ ] 3.3 Chạy **chỉ seed 0** và in train loss, val loss theo epoch

🛑 **DỪNG 4** — người dùng xem đường loss của seed 0 (giảm rồi phẳng, không NaN). Chờ "tiếp" rồi mới chạy các seed còn lại.

- [ ] 3.4 Chạy seed 1 và seed 2
- [ ] 3.5 `src/evaluate.py`: `reconstruction_score`, lưu score cho val, calib 1-3, test 0-3, đủ 3 seed, đặt tên `cnn_ae_seed{s}_...`
- [ ] 3.6 In AUC-ROC và AUC-PR của CNN-AE ở 0 HP; in score trung bình window fault so với normal

🛑 **DỪNG 5** — người dùng kiểm tra score fault > score normal ở 0 HP và đủ file score của 3 seed. Chờ "tiếp".

---

## Giai đoạn 4 — Setup1 và Setup2 (HUONG_DAN: Bước 4)

- [ ] 4.1 `src/make_tables.py`: `summarize(df, group_cols)` dùng chung; ngưỡng p99 từ val; xuất `results/tables/setup1.csv` và `setup2.csv` dạng mean ± std
- [ ] 4.2 In hai bảng; kiểm tra std khác 0 và số 0 HP của hai bảng khớp nhau

🛑 **DỪNG 6** — người dùng đọc kết quả RQ1 (CNN-AE vs Isolation Forest ở 0 HP) và RQ2 (FPR tăng thế nào khi đổi tải). Chờ "tiếp".

---

## Giai đoạn 5 — Setup3 (HUONG_DAN: Bước 5)

- [ ] 5.1 Lưu model Isolation Forest đã fit (joblib) hoặc fit lại cùng seed; viết `score_any(model_name, seed, X)` cho cả hai model
- [ ] 5.2 `src/noise.py`: `add_noise(X_raw, snr_db, seed)`; chạy SNR 20/10/5, xuất `setup3_noise.csv`
- [ ] 5.3 `src/thresholds.py`: `thr_p99`, `thr_mean3s`, `thr_pot`, `thr_per_condition`, `thr_oracle`
- [ ] 5.4 Chạy thử POT trên val score và in `N_t`, xi, sigma, ngưỡng; so với p99

🛑 **DỪNG 7** — người dùng xem POT: `N_t` đủ lớn (khoảng 10 trở lên)? ngưỡng cùng cỡ p99, không NaN? Nếu không: hỏi hạ `init_q` hoặc dùng val stride 256. Chờ "tiếp".

- [ ] 5.5 Xuất `setup3_threshold.csv` (4 chiến lược, CNN-AE là chính, thêm Isolation Forest nếu kịp)
- [ ] 5.6 Ablation chuẩn hóa theo từng tải (Bước 5c) → `setup3_ablation.csv`
- [ ] 5.7 Chạy ba mục "Tự kiểm tra" của Bước 5 (SNR thấp thì AUC-PR giảm; POT hợp lý; Oracle F1 >= các ngưỡng khác)

🛑 **DỪNG 8** — người dùng đọc kết quả RQ3 (nhiễu, ngưỡng, chuẩn hóa). Chờ "tiếp".

---

## Giai đoạn 6 — Hình và phân tích (HUONG_DAN: Bước 6)

- [ ] 6.1 `src/plots.py`: 6 hình (loss, phân bố score theo tải, FPR theo tải, F1 theo SNR, F1 theo chiến lược ngưỡng, gốc vs tái tạo), lưu `results/figures/`
- [ ] 6.2 Chọn 4-6 ca lỗi (fault bị bỏ sót; normal báo nhầm ở 3 HP; ca được cứu nhờ chuẩn hóa theo điều kiện), vẽ gốc/tái tạo/sai số

🛑 **DỪNG 9** — người dùng xem hình, đối chiếu số liệu hình với CSV. Chờ "tiếp".

---

## Giai đoạn 7 — Đóng gói (HUONG_DAN: Bước 7-8)

- [ ] 7.1 `scripts/run_all.sh` (bắt đầu từ `prepare`, `set -e`), mỗi file có `main()`
- [ ] 7.2 `README.md` đủ 7 mục (tiêu đề `DL2026-Group18-Project38`)
- [ ] 7.3 Hoàn thiện `DATA.md` (thêm sampling rate cuối cùng, quyết định ghi ở dưới)
- [ ] 7.4 Clone repo vào thư mục trống khác, làm theo README, so `setup1.csv` mới với bản cũ

🛑 **DỪNG 10 (cuối)** — người dùng chạy checklist nộp (Bước 8b): số liệu report khớp CSV, repo public/cấp quyền, đủ file.

---

## Ghi chú quyết định (Claude Code điền khi gặp)

- Môi trường: dùng Python 3.12 (.venv), torch bản CPU, theo HUONG_DAN Bước 0 (máy mặc định là 3.14, không dùng).
- .gitignore thêm `*.mat`, `*.npz`, `*.pt` vì dữ liệu thô đang nằm ở `src/data/data/raw/` (không phải `data/raw/`), pattern `data/raw/` không chặn được.
- Dữ liệu thô nằm ở `src/data/data/raw/` -> đã chuyển về `data/raw/` đúng như HUONG_DAN.
- inspect_raw: tên biến khớp số file; mỗi file một biến `_DE_time`, TRỪ `99.mat` có thêm `X098_DE_time` (trùng 98.mat) -> `load_de` chọn `X099_DE_time`. Không lệch tần số: normal 20-40 s, fault ~10 s đều hợp lý với 12 kHz -> không resample.
- Val/calib dùng stride 256 (`threshold_stride` trong config) vì stride 1024 chỉ ra ~34-47 window. Test giữ stride 1024. Hệ quả: normal test 0 HP chỉ có 35 window nên FPR ở 0 HP thô (bước 1/35 ~ 2,9%).
- Mean/std chuẩn hóa tính từ tín hiệu train 0 HP: mean=0.01255, std=0.07242.
- Làm hết TODO một mạch theo yêu cầu người dùng ("làm hết cái todo luôn"), các mốc 🛑 chỉ in kết quả, không chờ.
