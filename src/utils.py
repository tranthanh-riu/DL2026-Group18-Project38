import random

import numpy as np
import torch
import yaml


def load_config(path="configs/default.yaml"):
    # Đọc file yaml thành dict để mọi script dùng chung một nguồn tham số
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    # Đặt cùng một seed cho cả 3 thư viện để kết quả lặp lại được
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
