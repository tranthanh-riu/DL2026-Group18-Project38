import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import precision_recall_curve

from src.utils import load_config


def thr_p99(val):
    # Percentile 99 của score normal trên val
    return np.percentile(val, 99)


def thr_mean3s(val):
    # mean + 3 độ lệch chuẩn
    return val.mean() + 3 * val.std()


def pot_details(val, init_q, q):
    # POT: lấy phần vượt ngưỡng khởi đầu t, fit Generalized Pareto, ngoại suy tới xác suất rủi ro q
    t = np.quantile(val, init_q)
    excess = val[val > t] - t
    num_excess = len(excess)
    n = len(val)
    xi, _, sigma = genpareto.fit(excess, floc=0)
    ratio = q * n / num_excess
    if abs(xi) < 1e-6:
        # Dạng giới hạn khi xi -> 0
        z_q = t - sigma * np.log(ratio)
    else:
        z_q = t + (sigma / xi) * (ratio ** (-xi) - 1)
    return {"t": t, "N_t": num_excess, "xi": xi, "sigma": sigma, "z_q": z_q}


def thr_pot(val, init_q, q):
    return pot_details(val, init_q, q)["z_q"]


def thr_per_condition(calib_L):
    # Ngưỡng p99 từ calib của CHÍNH tải đó (với 0 HP truyền val vào)
    return np.percentile(calib_L, 99)


def thr_oracle(y, score):
    # Cận trên: ngưỡng cho F1 cao nhất trên test (nhìn nhãn test nên chỉ để tham khảo)
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_threshold = thresholds[np.argmax(f1_values)]
    # precision_recall_curve dùng "score >= thr" còn at_threshold dùng "score > thr"
    # -> hạ ngưỡng xuống một nấc nhỏ để hai cách khớp nhau
    return np.nextafter(np.float64(best_threshold), -np.inf)


def main():
    config = load_config()
    # Test nhanh bằng score giả (đuôi nặng)
    val = np.random.default_rng(0).exponential(1.0, size=500)
    print("p99 =", round(thr_p99(val), 3), "| mean+3s =", round(thr_mean3s(val), 3))
    print("POT:", pot_details(val, config["pot"]["init_quantile"], config["pot"]["q"]))


if __name__ == "__main__":
    main()
