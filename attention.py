import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int = 128):
        super().__init__()
        self.d_k = d_k

    def forward(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            padding_mask: torch.Tensor | None = None,
            sequence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        :param padding_mask: mask that blocks out padding tokens
        :param sequence_mask: mask that allows to see only the previous and current tokens
        :returns: attention weights
        """
        attn_scores = Q.matmul(K.transpose(2, 1)) / np.sqrt(self.d_k)  # (batch_size, seq_len, seq_len)

        if padding_mask is not None:
            attn_scores = attn_scores.masked_fill(padding_mask == 0, float('-inf'))

        if sequence_mask is not None:
            # masking out (setting to −∞) all values in the input of the softmax
            # which correspond to illegal connections
            attn_scores = attn_scores.masked_fill(sequence_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        output = torch.bmm(attn_weights, V)  # (batch_size, seq_len, d_v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8, d_model=512, d_k=128, d_v=256):
        super().__init__()
        self.n_heads = n_heads
        self.Q_proj = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(n_heads)])
        self.K_proj = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(n_heads)])
        self.V_proj = nn.ModuleList([nn.Linear(d_model, d_v) for _ in range(n_heads)])
        self.attn_layers = nn.ModuleList([ScaledDotProductAttention(d_k) for _ in range(n_heads)])
        self.final_fc = nn.Linear(d_v * n_heads, d_model)

    def forward(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            padding_mask: torch.Tensor,
            sequence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_outputs = []  # tensors of shape (batch_size, seq_len, d_v)
        for i in range(self.n_heads):
            Q_proj = self.Q_proj[i](Q)  # (batch_size, seq_len, d_k)
            K_proj = self.K_proj[i](K)  # (batch_size, seq_len, d_k)
            V_proj = self.V_proj[i](V)  # (batch_size, seq_len, d_v)
            attn_outputs.append(self.attn_layers[i](Q_proj, K_proj, V_proj, padding_mask, sequence_mask))
        output = torch.cat(attn_outputs, dim=-1)  # (batch_size, seq_len, d_v * n_heads)
        output = self.final_fc(output)  # (batch_size, seq_len, d_model)
        return output
