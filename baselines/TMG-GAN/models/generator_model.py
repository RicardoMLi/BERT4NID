import torch
from torch import nn

from utils import init_weights


class GeneratorModel(nn.Module):
    def __init__(self, z_size: int, feature_num: int):
        super().__init__()
        self.z_size = z_size
        self.main_model = nn.Sequential(
            # z_size * 1 * 1
            nn.Linear(z_size, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            # 64 * 4 * 4
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            # 32 * 8 * 8
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            # 16 * 16 * 16
        )
        self.hidden_status: torch.Tensor = None
        self.last_layer = nn.Sequential(
            nn.Linear(16, feature_num),
            nn.BatchNorm1d(feature_num),
            nn.LeakyReLU(0.2),
            # 3 * 32 * 32
            nn.Tanh()
        )
        self.apply(init_weights)

    def generate_samples(self, num: int) -> torch.Tensor:
        z = torch.randn(num, self.z_size, device='cuda')
        return self.forward(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main_model(x)
        self.hidden_status = x
        return self.last_layer(x)

