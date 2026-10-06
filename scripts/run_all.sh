#!/usr/bin/env bash
# data -> train -> evaluate -> bảng/hình
# Dữ liệu thô (16 file .mat) phải có sẵn trong data/raw/ (xem DATA.md), không có bước tải.
# Chạy từ thư mục gốc repo: bash scripts/run_all.sh
set -e

python -m src.data.prepare      # Bước 1: cắt window, chia split, chuẩn hóa -> data/processed/
python -m src.models.iforest    # Bước 2: Isolation Forest, lưu score
python -m src.train             # Bước 3: train CNN-AE 3 seed
python -m src.evaluate          # Bước 3: score CNN-AE
python -m src.make_tables       # Bước 4: setup1.csv, setup2.csv (phải chạy TRƯỚC setup3 vì ablation đọc per_seed_results.csv)
python -m src.setup3            # Bước 5: nhiễu, 4 ngưỡng, ablation
python -m src.plots             # Bước 6: hình + ca lỗi
