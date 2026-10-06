import json
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.utils import load_config

RAW_FOLDER = "data/raw"
PROCESSED_FOLDER = "data/processed"

# Ánh xạ tải (HP) -> số file .mat (HUONG_DAN Bước 1a)
FILES = {
    0: {"normal": 97,  "IR": 105, "B": 118, "OR": 130},
    1: {"normal": 98,  "IR": 106, "B": 119, "OR": 131},
    2: {"normal": 99,  "IR": 107, "B": 120, "OR": 132},
    3: {"normal": 100, "IR": 108, "B": 121, "OR": 133},
}


def load_de(number):
    # Đọc tín hiệu cảm biến Drive End. Ưu tiên key đúng số file vì 99.mat
    # chứa thêm biến X098_DE_time (bản sao của 98.mat)
    mat = loadmat(os.path.join(RAW_FOLDER, f"{number}.mat"))
    de_keys = []
    for key in mat.keys():
        if key.endswith("_DE_time"):
            de_keys.append(key)

    wanted_key = f"X{number:03d}_DE_time"
    if wanted_key in de_keys:
        key = wanted_key
    else:
        key = de_keys[0]
        print(f"CANH BAO: file {number} khong co {wanted_key}, dung {key}")
    return mat[key].ravel().astype(np.float32)


def make_windows(sig, size, stride):
    # Cắt tín hiệu 1 chiều thành các window (N, size); .copy() để không còn là view của sig
    return np.lib.stride_tricks.sliding_window_view(sig, size)[::stride].copy()


def split_normal_0hp(sig, split):
    # Cắt TÍN HIỆU trước (theo thời gian), cắt window sau, để không window nào vắt qua 2 split
    n = len(sig)
    train_end = int(split[0] * n)
    val_end = int((split[0] + split[1]) * n)
    return sig[:train_end], sig[train_end:val_end], sig[val_end:]


def build_windows(signal_list, size, stride):
    # signal_list: danh sách (tín hiệu, nhãn, loại lỗi). Trả về X_raw (N, size), y (N,), fault_type (N,)
    all_windows = []
    all_labels = []
    all_types = []
    for sig, label, fault_type in signal_list:
        windows = make_windows(sig, size, stride)
        all_windows.append(windows)
        all_labels.append(np.full(len(windows), label, dtype=np.int64))
        all_types.append(np.full(len(windows), fault_type))
    return np.concatenate(all_windows), np.concatenate(all_labels), np.concatenate(all_types)


def save_split(name, signal_list, size, stride, mean, std, table_rows, load):
    # Chuẩn hóa bằng mean/std TOÀN CỤC của train 0 HP rồi lưu .npz (X, X_raw, y, fault_type)
    X_raw, y, fault_type = build_windows(signal_list, size, stride)
    X = ((X_raw - mean) / std).astype(np.float32)
    path = os.path.join(PROCESSED_FOLDER, f"{name}.npz")
    np.savez(path, X=X, X_raw=X_raw, y=y, fault_type=fault_type)
    for one_type in np.unique(fault_type):
        count = int((fault_type == one_type).sum())
        table_rows.append({"load": load, "split": name.split("_")[1], "fault_type": str(one_type), "n_windows": count})
    print(f"da luu {path}: X {X.shape}")


def prepare_load(load, config, mean, std, train_val_signals, table_rows):
    # Tạo các file của một tải: test (normal + 3 loại fault) và calib (nếu tải 1-3)
    size = config["window"]
    normal_signal = load_de(FILES[load]["normal"])
    fault_list = []
    for fault_type in ["IR", "B", "OR"]:
        fault_list.append((load_de(FILES[load][fault_type]), 1, fault_type))

    if load == 0:
        normal_test = train_val_signals[2]
    else:
        calib_end = int(config["calib_frac"] * len(normal_signal))
        normal_test = normal_signal[calib_end:]
        calib_list = [(normal_signal[:calib_end], 0, "normal")]
        save_split(f"load{load}_calib", calib_list, size, config["threshold_stride"], mean, std, table_rows, load)

    test_list = [(normal_test, 0, "normal")] + fault_list
    save_split(f"load{load}_test", test_list, size, config["eval_stride"], mean, std, table_rows, load)


def check_outputs():
    # Tự kiểm tra của Bước 1: shape, NaN, mean/std của train, y có đủ 0 và 1
    train = np.load(os.path.join(PROCESSED_FOLDER, "load0_train.npz"))
    print("\n--- Tu kiem tra ---")
    print("load0_train X.shape:", train["X"].shape, "| co NaN:", bool(np.isnan(train["X"]).any()))
    print("load0_train X mean = %.4f, std = %.4f (ky vong ~0 va ~1)" % (train["X"].mean(), train["X"].std()))
    for load in [0, 1, 2, 3]:
        test = np.load(os.path.join(PROCESSED_FOLDER, f"load{load}_test.npz"))
        print(f"load{load}_test: gia tri cua y = {np.unique(test['y'])}, NaN = {bool(np.isnan(test['X']).any())}")


def main():
    config = load_config()
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    size = config["window"]
    table_rows = []

    # Tín hiệu normal 0 HP chia 70/15/15 theo thời gian
    normal_0hp = load_de(FILES[0]["normal"])
    train_signal, val_signal, test_signal = split_normal_0hp(normal_0hp, config["split"])
    print("do dai tin hieu 0 HP: train", len(train_signal), "val", len(val_signal), "test", len(test_signal))

    # mean/std là 2 số vô hướng, chỉ tính từ train 0 HP để val/test không rò rỉ vào chuẩn hóa
    mean = float(train_signal.mean())
    std = float(train_signal.std())
    with open(os.path.join(PROCESSED_FOLDER, "norm_global.json"), "w") as f:
        json.dump({"mean": mean, "std": std}, f)
    print("norm_global: mean = %.5f, std = %.5f" % (mean, std))

    save_split("load0_train", [(train_signal, 0, "normal")], size, config["train_stride"], mean, std, table_rows, 0)
    save_split("load0_val", [(val_signal, 0, "normal")], size, config["threshold_stride"], mean, std, table_rows, 0)
    for load in [0, 1, 2, 3]:
        prepare_load(load, config, mean, std, (train_signal, val_signal, test_signal), table_rows)

    table = pd.DataFrame(table_rows)
    print("\nSo window theo tai x split x fault_type:")
    print(table.groupby(["load", "split", "fault_type"]).n_windows.sum().to_string())
    check_outputs()


if __name__ == "__main__":
    main()
