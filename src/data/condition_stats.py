import os
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")


def calculate_stats(load):
    path = os.path.join(PROCESSED_DIR, f"load{load}_test.npz")

    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    normal_X = X[y == 0]

    mean = float(np.mean(normal_X))
    std = float(np.std(normal_X))

    return mean, std, len(normal_X)


def main():
    stats = {}

    for load in range(4):
        mean, std, count = calculate_stats(load)

        stats[f"load{load}_mean"] = mean
        stats[f"load{load}_std"] = std

        print(
            f"Load {load}: "
            f"normal_windows={count}, "
            f"mean={mean:.10f}, "
            f"std={std:.10f}"
        )

    output_path = os.path.join(PROCESSED_DIR, "condition_stats.npz")

    np.savez(
        output_path,
        **stats
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()