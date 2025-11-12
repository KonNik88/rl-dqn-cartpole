from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class NatureCNN(nn.Module):
    """
    Atari Nature-CNN.
    Ожидает вход уже в [0,1] (float32). Деления на 255 внутри НЕТ.
    in_channels: число кадров в стеке (обычно 4)
    act_dim: размер дискретного action space
    dueling: включить Dueling head (V(s) + A(s,a) - mean_a A(s,a))
    """
    def __init__(self, in_channels: int = 4, act_dim: int = 6, dueling: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),          nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True)
        )
        self.dueling = bool(dueling)
        if self.dueling:
            self.V = nn.Linear(512, 1)
            self.A = nn.Linear(512, act_dim)
        else:
            self.Q = nn.Linear(512, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.conv(x)
        x = self.fc(x)
        if self.dueling:
            V = self.V(x)                 # [B,1]
            A = self.A(x)                 # [B,A]
            Q = V + A - A.mean(dim=1, keepdim=True)
            return Q
        else:
            return self.Q(x)
