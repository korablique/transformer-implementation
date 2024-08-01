import torch
from torch import nn
import torch.nn.functional as F

from transformer_implementation.attention import MultiHeadAttention
from transformer_implementation.feed_forward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, n_heads=8, d_model=512, d_k=128, d_v=256, d_ff=2048):
        super().__init__()

        self.enc_dec_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.masked_self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(
            self,
            output_embs: torch.Tensor,  # decoder outputs
            encoder_output: torch.Tensor,
            inputs_padding_mask: torch.Tensor | None = None,
            outputs_padding_mask: torch.Tensor | None = None,
            sequence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # masked self attention sublayer
        masked_self_attn = self.masked_self_attn(output_embs, output_embs, output_embs, outputs_padding_mask, sequence_mask)
        masked_self_attn += output_embs
        masked_self_attn = F.dropout(masked_self_attn, p=0.1, training=self.training)
        masked_self_attn = self.layer_norm1(masked_self_attn)

        # encoder-decoder attention sublayer
        enc_dec_attn = self.enc_dec_attn(masked_self_attn, encoder_output, encoder_output, inputs_padding_mask)
        enc_dec_attn += masked_self_attn
        enc_dec_attn = F.dropout(enc_dec_attn, p=0.1, training=self.training)
        enc_dec_attn = self.layer_norm2(enc_dec_attn)

        # feed forward sublayer
        output = enc_dec_attn + self.ff(enc_dec_attn)
        output = F.dropout(output, p=0.1, training=self.training)
        output = self.layer_norm3(output)

        return output
