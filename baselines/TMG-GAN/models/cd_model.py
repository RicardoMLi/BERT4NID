import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from utils import init_weights


class CDModel(nn.Module):
    def __init__(self, in_features: int, label_num: int):
        super().__init__()
        self.main_model = nn.Sequential(
            # 3 * 32 * 32
            spectral_norm(nn.Linear(in_features, 1024)),
            nn.LeakyReLU(0.2),
            # 16 * 16 * 16
            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2),
            # 32 * 8 * 8
            spectral_norm(nn.Linear(512, 128)),
            nn.LeakyReLU(0.2),
        )
        self.hidden_status: torch.Tensor = None
        self.c_last_layer = nn.Sequential(
            nn.Linear(128, label_num),
            nn.Softmax(dim=1),
        )
        self.d_last_layer = nn.Sequential(
            spectral_norm(nn.Linear(128, 1)),
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.main_model(x)
        self.hidden_status = x
        return self.d_last_layer(x), self.c_last_layer(x)
