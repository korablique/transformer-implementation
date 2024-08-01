import numpy as np
import torch
from torch import nn

from transformer_implementation.decoder import DecoderLayer
from transformer_implementation.encoder import EncoderLayer
from transformer_implementation.positional_encoding import PositionalEncoding
from transformer_implementation.vocab import Vocab


class Transformer(nn.Module):
    def __init__(self, inp_vocab: Vocab, out_vocab: Vocab,
                 d_model=512, d_k=128, d_v=256, d_ff=2048, n_layers=6, n_heads=8, max_seq_len=1000):
        """
        :param inp_vocab: vocabulary of input tokens
        :param out_vocab: vocabulary of output tokens
        :param d_model: inner model tensors dimension
        :param d_k: dimension of keys and queries
        :param d_v: dimension of values
        :param d_ff: dimension of feed forward layers
        :param n_layers: number of encoder layers and decoder.py layers
        :param n_heads: number of heads in multi-head attention blocks
        :param max_seq_len: max possible tokens sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.inp_vocab = inp_vocab
        self.out_vocab = out_vocab

        self.input_emb = nn.Embedding(len(inp_vocab), d_model)
        self.output_emb = nn.Embedding(len(out_vocab), d_model)
        self.pos_enc = PositionalEncoding(max_seq_len, d_model)
        self.dropout = nn.Dropout(p=0.1)

        self.encoder = nn.ModuleList([EncoderLayer(n_heads, d_model, d_k, d_v, d_ff) for _ in range(n_layers)])

        self.decoder = nn.ModuleList([DecoderLayer(n_heads, d_model, d_k, d_v, d_ff) for _ in range(n_layers)])

        self.fc = nn.Linear(d_model, len(out_vocab))
        self.fc.weight = self.output_emb.weight  # share the same weight matrix between embedding layers and the pre-softmax linear transformation

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: input sequence of token indices (batch_size, seq_len)
        :param outputs: target sequence on train and start token index on inference (batch_size, seq_len)
        """
        device = next(self.fc.parameters()).device

        input_embs = self.input_emb(inputs) * np.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        input_embs = self.dropout(self.pos_enc(input_embs))  # add positional embeddings

        inputs_padding_mask = self.inp_vocab.compute_padding_mask(inputs).unsqueeze(1)

        encoder_output = input_embs.clone()
        for layer in self.encoder:
            encoder_output = layer(encoder_output, inputs_padding_mask)

        # decoding
        output_embs = self.output_emb(outputs)
        output_embs = self.dropout(self.pos_enc(output_embs))
        outputs_padding_mask = self.out_vocab.compute_padding_mask(outputs).unsqueeze(1)

        seq_len = output_embs.size(1)
        sequence_mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)  # (seq_len, seq_len)

        decoder_output = output_embs.clone()
        for layer in self.decoder:
            decoder_output = layer(decoder_output, encoder_output, inputs_padding_mask, outputs_padding_mask, sequence_mask)

        output_logits = self.fc(decoder_output)  # (batch_size, d_model)
        return output_logits

    def translate(self, inp_text: str) -> str:
        """
        inference function
        :param inp_text: input string processed with BPE
        :return: translated string
        """
        device = next(self.fc.parameters()).device

        with torch.no_grad():
            # text to tensor
            inp_matrix = self.inp_vocab.to_matrix([inp_text]).to(device)

            # encode input text
            input_embs = self.input_emb(inp_matrix) * np.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
            input_embs = self.pos_enc(input_embs)  # add positional embeddings

            encoder_output = input_embs.clone()
            for layer in self.encoder:
                encoder_output = layer(encoder_output)

            # decoding loop
            translated_tokens = [self.out_vocab.bos_idx]  # initialize the target sequence with the start token
            for _ in range(self.max_seq_len):
                out_matrix = torch.LongTensor(translated_tokens).unsqueeze(0).to(device)
                output_embs = self.output_emb(out_matrix)
                output_embs = self.pos_enc(output_embs)

                decoder_output = output_embs.clone()
                for layer in self.decoder:
                    decoder_output = layer(decoder_output, encoder_output)

                # predict next token
                next_token_probs = torch.softmax(self.fc(decoder_output)[0, -1, :], dim=-1).cpu().numpy()
                next_token_idx = np.random.choice(len(self.out_vocab), p=next_token_probs)
                if next_token_idx == self.out_vocab.eos_idx:
                    break
                translated_tokens.append(next_token_idx)
            return ' '.join([self.out_vocab.vocab[token_idx] for token_idx in translated_tokens])
