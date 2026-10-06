import torch
import torch.nn as nn


class CNNAE(nn.Module):
    # Autoencoder 1D: nén (1, 1024) -> (64, 128) rồi dựng lại (1, 1024)
    def __init__(self):
        super().__init__()
        # Mọi lớp dùng kernel 4, stride 2, padding 1 -> độ dài giảm/tăng đúng một nửa/gấp đôi
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),    # 1024 -> 512
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),   # 512 -> 256
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),   # 256 -> 128
        )
        # Lớp cuối của decoder KHÔNG có ReLU vì tín hiệu chuẩn hóa có giá trị âm
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 128 -> 256
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 256 -> 512
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),               # 512 -> 1024
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def main():
    model = CNNAE()
    x = torch.randn(8, 1, 1024)
    print("input:", x.shape, "-> latent:", model.encoder(x).shape, "-> output:", model(x).shape)
    num_params = sum(p.numel() for p in model.parameters())
    print("so tham so:", num_params)


if __name__ == "__main__":
    main()
