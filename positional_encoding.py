import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int = 512, n: int = 10000):
        """
        :param max_seq_len: max possible sequence length
        :param d_model:
        :param n: a constant
        """
        super().__init__()
        self.d_model = d_model
        self.register_buffer('pe', self._generate_pe(max_seq_len, d_model, n))

    @staticmethod
    def _generate_pe(max_seq_len: int, d_model: int, n: int) -> torch.Tensor:
        pe = np.zeros((max_seq_len, d_model))
        token_index = np.arange(max_seq_len)[:, np.newaxis]  # (max_seq_len, 1)
        col_index = np.arange(d_model // 2)
        divider = n ** (2 * col_index / d_model)

        pe[:, 0::2] = np.sin(token_index / divider)
        pe[:, 1::2] = np.cos(token_index / divider)
        return torch.tensor(pe, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input embeddings (batch_size, seq_len, d_model)
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]
