import torch
from torch import nn
import torch.nn.functional as F

from transformer_implementation.attention import MultiHeadAttention
from transformer_implementation.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, n_heads=8, d_model=512, d_k=128, d_v=256, d_ff=2048):
        super().__init__()

        # self attention
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, input_embs: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param input_embs: input embeddings (batch_size, seq_len, d_model)
        :param padding_mask: mask that blocks out padding tokens
        :return:
        """
        # self attention sublayer
        output = self.self_attn(input_embs, input_embs, input_embs, padding_mask)
        output += input_embs  # add residual connection
        output = F.dropout(output, p=0.1)
        output = self.layer_norm1(output)

        # feed forward sublayer
        output = output + self.ff(output)
        output = F.dropout(output, p=0.1)
        output = self.layer_norm2(output)

        return output
