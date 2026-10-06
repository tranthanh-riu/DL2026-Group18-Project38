# DATA.md — CWRU Bearing Dataset (Đề #38)

## Nguồn
- Normal baseline: https://engineering.case.edu/bearingdatacenter/normal-baseline-data
- 12k Drive End fault data: https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
- Mẫu URL file: `https://engineering.case.edu/sites/default/files/<số>.mat`
- Ngày tải: 06/10/2026 (tải thủ công vào `data/raw/`, không có script tải)

## 16 file sử dụng
| Loại | 0 HP | 1 HP | 2 HP | 3 HP |
| --- | --- | --- | --- | --- |
| Normal | 97 | 98 | 99 | 100 |
| Inner race 0,007" (IR) | 105 | 106 | 107 | 108 |
| Ball 0,007" (B) | 118 | 119 | 120 | 121 |
| Outer race @6:00 0,007" (OR) | 130 | 131 | 132 | 133 |

Dùng biến `X<số>_DE_time` (cảm biến Drive End). Lưu ý: `99.mat` chứa thêm biến `X098_DE_time`
(bản sao của 98.mat); code chọn đúng `X099_DE_time`.

## Sampling rate
Toàn bộ dùng 12 kHz (nhóm 12k Drive End; file normal dài 20-40 s, file fault khoảng 10 s,
nhất quán với 12 kHz). Không cần resample. (Xem "Ghi chú quyết định" trong docs/TODO.md.)

## Tiền xử lý và chia split
- Window 1024 mẫu. Cắt tín hiệu theo thời gian trước, cắt window sau (không window nào vắt qua hai split).
- Normal 0 HP: 70% đầu train (stride 512), 15% val, 15% test.
- Normal 1/2/3 HP: 10% đầu calib, 90% còn lại test.
- Fault: toàn bộ vào test của tải tương ứng, nhãn 1.
- Test: stride 1024. Val và calib: stride 256 (chỉ để ước lượng ngưỡng, vì stride 1024 chỉ cho khoảng 35-47 window;
  các window vẫn nằm trọn trong đoạn val/calib nên không rò rỉ).
- Chuẩn hóa toàn cục: mean, std là hai số vô hướng tính từ train 0 HP (`data/processed/norm_global.json`),
  áp cho mọi split, mọi tải. `X_raw` (chưa chuẩn hóa) cũng được lưu để thêm nhiễu và làm ablation.

## Số window (load x split x fault_type)
| Tải | Split | normal | IR | B | OR |
| --- | --- | --- | --- | --- | --- |
| 0 | train | 332 | | | |
| 0 | val | 139 | | | |
| 0 | test | 35 | 118 | 119 | 119 |
| 1 | calib | 186 | | | |
| 1 | test | 425 | 119 | 118 | 119 |
| 2 | calib | 186 | | | |
| 2 | test | 426 | 119 | 118 | 118 |
| 3 | calib | 186 | | | |
| 3 | test | 426 | 120 | 118 | 119 |

## Tái tạo
```bash
python -m src.data.prepare
```
