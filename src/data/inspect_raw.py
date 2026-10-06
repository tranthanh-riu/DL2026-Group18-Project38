import os

from scipy.io import loadmat

# 16 file cần kiểm tra (số file theo bảng ở HUONG_DAN Bước 1a)
FILE_NUMBERS = [97, 98, 99, 100, 105, 106, 107, 108, 118, 119, 120, 121, 130, 131, 132, 133]
RAW_FOLDER = "data/raw"


def inspect_one_file(number):
    # Mở một file .mat và in các biến kết thúc bằng _DE_time (cảm biến Drive End)
    path = os.path.join(RAW_FOLDER, f"{number}.mat")
    mat = loadmat(path)
    de_keys = []
    for key in mat.keys():
        if key.endswith("_DE_time"):
            de_keys.append(key)

    expected_key = f"X{number:03d}_DE_time"
    for key in de_keys:
        length = mat[key].size
        # Chia cho 12000 và 48000 để biết thời lượng nếu file là 12 kHz hay 48 kHz
        seconds_12k = length / 12000
        seconds_48k = length / 48000
        match_text = "khop" if key == expected_key else "LECH so file"
        print(f"{number:>4} | {key:<16} | len={length:>7} | 12k={seconds_12k:6.2f}s | "
              f"48k={seconds_48k:6.2f}s | {match_text} | so bien DE={len(de_keys)}")
    if len(de_keys) == 0:
        print(f"{number:>4} | KHONG co bien _DE_time. Cac key: {list(mat.keys())}")


def main():
    print("so file | ten bien | do dai | thoi luong neu 12 kHz / 48 kHz | ten khop?")
    for number in FILE_NUMBERS:
        inspect_one_file(number)


if __name__ == "__main__":
    main()
