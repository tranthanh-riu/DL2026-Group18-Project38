from scipy.io import loadmat
import numpy as np
from pathlib import Path


WINDOW_SIZE = 1024
TRAIN_STRIDE = 512


FILE_INFO = {
    "97.mat": {
        "load": 0,
        "label": 0,
        "fault_type": "normal",
        "fault_size": 0.0
    },
    "98.mat": {
        "load": 1,
        "label": 0,
        "fault_type": "normal",
        "fault_size": 0.0
    },
    "99.mat": {
        "load": 2,
        "label": 0,
        "fault_type": "normal",
        "fault_size": 0.0
    },
    "100.mat": {
        "load": 3,
        "label": 0,
        "fault_type": "normal",
        "fault_size": 0.0
    },

    "105.mat": {
        "load": 0,
        "label": 1,
        "fault_type": "inner_race",
        "fault_size": 0.007
    },
    "106.mat": {
        "load": 1,
        "label": 1,
        "fault_type": "inner_race",
        "fault_size": 0.007
    },
    "107.mat": {
        "load": 2,
        "label": 1,
        "fault_type": "inner_race",
        "fault_size": 0.007
    },
    "108.mat": {
        "load": 3,
        "label": 1,
        "fault_type": "inner_race",
        "fault_size": 0.007
    },

    "118.mat": {
        "load": 0,
        "label": 1,
        "fault_type": "ball",
        "fault_size": 0.007
    },
    "119.mat": {
        "load": 1,
        "label": 1,
        "fault_type": "ball",
        "fault_size": 0.007
    },
    "120.mat": {
        "load": 2,
        "label": 1,
        "fault_type": "ball",
        "fault_size": 0.007
    },
    "121.mat": {
        "load": 3,
        "label": 1,
        "fault_type": "ball",
        "fault_size": 0.007
    },

    "130.mat": {
        "load": 0,
        "label": 1,
        "fault_type": "outer_race_6",
        "fault_size": 0.007
    },
    "131.mat": {
        "load": 1,
        "label": 1,
        "fault_type": "outer_race_6",
        "fault_size": 0.007
    },
    "132.mat": {
        "load": 2,
        "label": 1,
        "fault_type": "outer_race_6",
        "fault_size": 0.007
    },
    "133.mat": {
        "load": 3,
        "label": 1,
        "fault_type": "outer_race_6",
        "fault_size": 0.007
    },
}


def load_de_signal(mat_file):
    data = loadmat(mat_file)

    file_name = Path(mat_file).name
    file_number = Path(mat_file).stem

    target_key = f"X{file_number.zfill(3)}_DE_time"

    if target_key not in data:
        raise ValueError(
            f"Không tìm thấy {target_key} trong {file_name}"
        )

    signal = data[target_key].flatten()

    return signal


def create_windows(
    signal,
    window_size=WINDOW_SIZE,
    stride=WINDOW_SIZE
):
    if len(signal) < window_size:
        return np.empty((0, window_size))

    starts = range(
        0,
        len(signal) - window_size + 1,
        stride
    )

    windows = np.array([
        signal[start:start + window_size]
        for start in starts
    ])

    return windows


def split_normal_0hp(signal):
    n = len(signal)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_signal = signal[:train_end]
    val_signal = signal[train_end:val_end]
    test_signal = signal[val_end:]

    train_windows = create_windows(
        train_signal,
        WINDOW_SIZE,
        TRAIN_STRIDE
    )

    val_windows = create_windows(
        val_signal,
        WINDOW_SIZE,
        WINDOW_SIZE
    )

    test_windows = create_windows(
        test_signal,
        WINDOW_SIZE,
        WINDOW_SIZE
    )

    return train_windows, val_windows, test_windows


def split_normal_other_load(signal):
    n = len(signal)

    calibration_end = int(n * 0.10)

    calibration_signal = signal[:calibration_end]
    test_signal = signal[calibration_end:]

    calibration_windows = create_windows(
        calibration_signal,
        WINDOW_SIZE,
        WINDOW_SIZE
    )

    test_windows = create_windows(
        test_signal,
        WINDOW_SIZE,
        WINDOW_SIZE
    )

    return calibration_windows, test_windows


def prepare_fault(signal):
    test_windows = create_windows(
        signal,
        WINDOW_SIZE,
        WINDOW_SIZE
    )

    return test_windows


def prepare_file(mat_file):
    file_name = Path(mat_file).name

    if file_name not in FILE_INFO:
        raise ValueError(
            f"File chưa được khai báo trong FILE_INFO: {file_name}"
        )

    info = FILE_INFO[file_name]

    signal = load_de_signal(mat_file)

    result = {
        "load": info["load"],
        "label": info["label"],
        "fault_type": info["fault_type"],
        "fault_size": info["fault_size"],
    }

    if info["label"] == 0 and info["load"] == 0:
        train, val, test = split_normal_0hp(signal)

        result["train"] = train
        result["val"] = val
        result["test"] = test

    elif info["label"] == 0:
        calibration, test = split_normal_other_load(signal)

        result["calibration"] = calibration
        result["test"] = test

    else:
        test = prepare_fault(signal)

        result["test"] = test

    return result


def calculate_train_stats(train_windows):
    mean = np.mean(train_windows)
    std = np.std(train_windows)

    return mean, std


def save_train_stats(mean, std, output_file):
    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        output_file,
        mean=mean,
        std=std
    )


def add_data_to_split(
    storage,
    load,
    split,
    X,
    y,
    fault_type,
    fault_size
):
    if X is None or len(X) == 0:
        return

    storage.setdefault(
        (load, split),
        {
            "X": [],
            "y": [],
            "fault_type": [],
            "fault_size": []
        }
    )

    n = len(X)

    storage[(load, split)]["X"].append(X)

    storage[(load, split)]["y"].append(
        np.full(n, y, dtype=np.int64)
    )

    storage[(load, split)]["fault_type"].extend(
        [fault_type] * n
    )

    storage[(load, split)]["fault_size"].extend(
        [fault_size] * n
    )


def export_processed_data(raw_dir, output_dir):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    storage = {}

    files = sorted(raw_dir.glob("*.mat"))

    for mat_file in files:
        file_name = mat_file.name

        if file_name not in FILE_INFO:
            continue

        info = FILE_INFO[file_name]

        result = prepare_file(mat_file)

        load = info["load"]
        label = info["label"]
        fault_type = info["fault_type"]
        fault_size = info["fault_size"]

        if label == 0 and load == 0:

            add_data_to_split(
                storage,
                load,
                "train",
                result["train"],
                label,
                fault_type,
                fault_size
            )

            add_data_to_split(
                storage,
                load,
                "val",
                result["val"],
                label,
                fault_type,
                fault_size
            )

            add_data_to_split(
                storage,
                load,
                "test",
                result["test"],
                label,
                fault_type,
                fault_size
            )

        elif label == 0:

            add_data_to_split(
                storage,
                load,
                "calibration",
                result["calibration"],
                label,
                fault_type,
                fault_size
            )

            add_data_to_split(
                storage,
                load,
                "test",
                result["test"],
                label,
                fault_type,
                fault_size
            )

        else:

            add_data_to_split(
                storage,
                load,
                "test",
                result["test"],
                label,
                fault_type,
                fault_size
            )

    for (load, split), data in sorted(storage.items()):

        X = np.concatenate(
            data["X"],
            axis=0
        )

        y = np.concatenate(
            data["y"],
            axis=0
        )

        fault_type = np.array(
            data["fault_type"],
            dtype="<U32"
        )

        fault_size = np.array(
            data["fault_size"],
            dtype=np.float32
        )

        output_file = (
            output_dir /
            f"load{load}_{split}.npz"
        )

        np.savez(
            output_file,
            X=X,
            y=y,
            fault_type=fault_type,
            fault_size=fault_size
        )

        print(
            f"Saved: {output_file} | "
            f"X={X.shape} | "
            f"y={y.shape}"
        )

    return storage


def print_window_summary(storage):
    print()
    print("=" * 80)
    print("WINDOW SUMMARY")
    print("=" * 80)

    print(
        f"{'Load':<8}"
        f"{'Split':<14}"
        f"{'Label':<8}"
        f"{'Fault type':<22}"
        f"{'Windows':<10}"
    )

    print("-" * 80)

    for (load, split), data in sorted(storage.items()):

        fault_counts = {}

        for fault_type in data["fault_type"]:

            if fault_type == "normal":
                label = 0
            else:
                label = 1

            key = (label, fault_type)

            fault_counts[key] = (
                fault_counts.get(key, 0) + 1
            )

        for (label, fault_type), count in sorted(
            fault_counts.items()
        ):

            print(
                f"{load:<8}"
                f"{split:<14}"
                f"{label:<8}"
                f"{fault_type:<22}"
                f"{count:<10}"
            )

    print("=" * 80)


if __name__ == "__main__":

    PROJECT_DIR = Path(__file__).resolve().parents[2]

    RAW_DIR = PROJECT_DIR.parent / "data" / "raw" / "CWRU"

    OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

    storage = export_processed_data(
        RAW_DIR,
        OUTPUT_DIR
    )

    print_window_summary(storage)