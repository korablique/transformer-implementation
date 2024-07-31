import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
