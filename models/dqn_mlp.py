from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256,256), layer_norm=True):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h)]
            if layer_norm: layers += [nn.LayerNorm(h)]
            layers += [nn.ReLU(inplace=True)]
            last = h
        self.body = nn.Sequential(*layers)
        self.V = nn.Linear(last, 1)
        self.A = nn.Linear(last, act_dim)

    def forward(self, x):
        x = self.body(x)
        V = self.V(x)
        A = self.A(x)
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q

class MLPQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256,256), layer_norm=True):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h)]
            if layer_norm: layers += [nn.LayerNorm(h)]
            layers += [nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, act_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
