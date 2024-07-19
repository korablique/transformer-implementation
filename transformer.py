import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from nltk.tokenize import WordPunctTokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
from tqdm import trange, tqdm

device = 'cpu'

tokenizer = WordPunctTokenizer()

bpe = {}
for lang in ['en', 'ru']:
    bpe[lang] = BPE(open('./bpe_rules.' + lang))


def tokenize(x: str) -> str:
    """
    :param x: text
    :return: tokenized text, tokens are separated by whitespace
    """
    return ' '.join(tokenizer.tokenize(x.lower()))


class Vocab:
    """
    Vocabulary of BPE tokens.
    """

    def __init__(self, texts: np.ndarray[str], bos: str = '<BOS>', eos: str = '<EOS>', unk: str = '<UNK>'):
        """
        :param texts: array of texts contained of bpe-tokens
        """
        self.bos, self.eos, self.unk = bos, eos, unk
        self.bos_idx, self.eos_idx, self.unk_idx = 0, 1, 2  # TODO hardcode

        unique_words = set(token for text in texts for token in text.split())
        self.vocab = [self.bos, self.eos, self.unk] + list(unique_words)  # index in the vocab == token index
        self.token_to_index = {t: i for i, t in enumerate(self.vocab)}

    def __len__(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: str) -> list[str]:
        """
        just splits the text by a whitespace and adds start and end tokens
        :param text: input text (assumed that bpe has already been applied) TODO
        :return: list of tokens
        """
        return [self.bos] + text.split() + [self.eos]

    def to_matrix(self, texts: np.ndarray[str] | list[str]) -> torch.Tensor:
        batch_size = len(texts)
        tokenized_texts = [self.tokenize(text) for text in texts]
        lengths = [len(text) for text in tokenized_texts]
        # create a matrix of size (batch_size, max_len),
        # where max_len is the length of the longest sequence in the batch
        matrix = np.full((batch_size, max(lengths)), fill_value=self.eos_idx)  # using EOS as padding token

        for i, tokens in enumerate(tokenized_texts):
            matrix[i, :lengths[i]] = [self.token_to_index.get(token, self.unk_idx) for token in tokens]

        return torch.from_numpy(matrix)

    def to_tokens(self, matrix: torch.Tensor) -> list:
        result = []
        for line in matrix:
            tokens = [self.vocab[token_idx] for token_idx in line if token_idx != self.eos_idx]
            result.append(' '.join(tokens))
        return result

    def compute_padding_mask(self, input_idxs: torch.Tensor) -> torch.Tensor:
        """
        compute a boolean mask that equals True until the first EOS including it
        :param input_idxs: (batch_size, seq_len)
        :returns: mask of shape (batch_size, seq_len)
        """
        return torch.cumsum(input_idxs == self.eos_idx, dim=-1) <= 1


class TranslationDataset(Dataset):
    def __init__(self, inp_texts: np.ndarray[str], out_texts: np.ndarray[str]):
        assert len(inp_texts) == len(out_texts), 'inp_texts and out_texts must be the same size'
        self.inp_texts = inp_texts
        self.out_texts = out_texts

    def __len__(self) -> int:
        return len(self.inp_texts)

    def __getitem__(self, idx: int):
        return self.inp_texts[idx], self.out_texts[idx]  # input string and output string


class FeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


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


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int = 128):
        super().__init__()
        self.d_k = d_k

    def forward(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            padding_mask: torch.Tensor,
            sequence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        :param padding_mask: mask that blocks out padding tokens
        :param sequence_mask: mask that allows to see only the previous and current tokens
        :returns: attention weights
        """
        attn_scores = Q.matmul(K.transpose(2, 1)) / np.sqrt(self.d_k)  # (batch_size, seq_len, seq_len)

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

    def forward(self, input_embs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        :param input_embs: input embeddings (batch_size, seq_len, d_model)
        :param padding_mask: mask that blocks out padding tokens
        :return:
        """
        # compute Q, K, V matrices (batch_size, seq_len, d_model)
        Q = self.query(input_embs)
        K = self.key(input_embs)
        V = self.value(input_embs)

        # self attention sublayer
        output = input_embs + self.self_attn(Q, K, V, padding_mask)  # add residual connection
        output = F.dropout(output, p=0.1)
        output = self.layer_norm1(output)

        # feed forward sublayer
        output = output + self.ff(output)
        output = F.dropout(output, p=0.1)
        output = self.layer_norm2(output)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_heads=8, d_model=512, d_k=128, d_v=256, d_ff=2048):
        super().__init__()

        self.query1 = nn.Linear(d_model, d_model)  # TODO refactoring needed
        self.key1 = nn.Linear(d_model, d_model)
        self.value1 = nn.Linear(d_model, d_model)

        self.query2 = nn.Linear(d_model, d_model)
        self.key2 = nn.Linear(d_model, d_model)
        self.value2 = nn.Linear(d_model, d_model)

        self.enc_dec_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.masked_self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(
            self,
            output_embs: torch.Tensor,
            encoder_output: torch.Tensor,
            inputs_padding_mask: torch.Tensor | None = None,
            outputs_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # masked self attention sublayer
        Q = self.query1(output_embs)  # (batch_size, seq_len, d_model)
        K = self.key1(output_embs)
        V = self.value1(output_embs)

        seq_len = output_embs.size(1)
        sequence_mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)  # (seq_len, seq_len)

        masked_self_attn = output_embs + self.masked_self_attn(Q, K, V, outputs_padding_mask, sequence_mask)
        masked_self_attn = F.dropout(masked_self_attn, p=0.1)
        masked_self_attn = self.layer_norm1(masked_self_attn)

        # encoder-decoder attention sublayer
        Q = self.query2(masked_self_attn)
        K = self.key2(encoder_output)
        V = self.value2(encoder_output)

        enc_dec_attn = output_embs + self.enc_dec_attn(Q, K, V, inputs_padding_mask)
        enc_dec_attn = F.dropout(enc_dec_attn, p=0.1)
        enc_dec_attn = self.layer_norm2(enc_dec_attn)

        # feed forward sublayer
        output = enc_dec_attn + self.ff(enc_dec_attn)
        output = F.dropout(output, p=0.1)
        output = self.layer_norm2(output)

        return output


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
        :param n_layers: number of encoder layers and decoder layers
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

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: input sequence of token indices (batch_size, seq_len)
        :param outputs: target sequence on train and start token index on inference (batch_size, seq_len)
        """
        input_embs = self.input_emb(inputs) * np.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        input_embs = self.dropout(self.pos_enc(input_embs))  # add positional embeddings
        inputs_padding_mask = self.inp_vocab.compute_padding_mask(inputs).unsqueeze(1)

        encoder_output = input_embs
        for layer in self.encoder:
            encoder_output = layer(encoder_output, inputs_padding_mask)

        # decoding
        output_embs = self.output_emb(outputs)
        output_embs = self.dropout(self.pos_enc(output_embs))
        outputs_padding_mask = self.out_vocab.compute_padding_mask(outputs).unsqueeze(1)

        decoder_output = output_embs
        for layer in self.decoder:
            decoder_output = layer(decoder_output, encoder_output, inputs_padding_mask, outputs_padding_mask)

        output_logits = self.fc(decoder_output)  # (batch_size, d_model)
        return output_logits

    def translate(self, inp_text: str) -> str:
        """
        inference function
        :param inp_text: input string
        :return: translated string
        """
        with torch.no_grad():
            # text to tensor
            inp_text_after_bpe = bpe['ru'].process_line(inp_text.lower())
            inp_matrix = self.inp_vocab.to_matrix([inp_text_after_bpe]).to(device)

            # encode input text
            input_embs = self.input_emb(inp_matrix) * np.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
            input_embs = self.pos_enc(input_embs)  # add positional embeddings
            inputs_padding_mask = self.inp_vocab.compute_padding_mask(inp_matrix)
            encoder_output = input_embs
            for layer in self.encoder:
                encoder_output = layer(encoder_output, inputs_padding_mask)

            # decoding loop
            translated_tokens = [self.out_vocab.bos_idx]  # initialize the target sequence with the start token
            for _ in range(self.max_seq_len):
                out_matrix = torch.LongTensor(translated_tokens).unsqueeze(0).to(device)
                output_embs = self.output_emb(out_matrix)
                output_embs = self.pos_enc(output_embs)
                outputs_padding_mask = self.out_vocab.compute_padding_mask(out_matrix)
                decoder_output = output_embs
                for layer in self.decoder:
                    decoder_output = layer(decoder_output, encoder_output, inputs_padding_mask, outputs_padding_mask)

                # predict next token
                next_token_idx = self.fc(decoder_output)[:, -1, :].argmax(dim=1).item()  # the most likely next token
                if next_token_idx == self.out_vocab.eos_idx:
                    break
                translated_tokens.append(next_token_idx)
            return ' '.join([self.out_vocab.vocab[token_idx] for token_idx in translated_tokens])


def compute_loss(model: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """
    :param model:
    :param inp: input tokens matrix
    :param out: reference tokens matrix
    """
    mask = model.out_vocab.compute_padding_mask(out)  # (batch_size, out_len), == True until the first EOS including it
    logits = model(inp, out)  # (batch_size, out_len, out_vocab_size)
    cross_entropy = F.cross_entropy(logits.transpose(1, 2), out, reduction='none')  # (batch_size, out_len)

    # average cross-entropy over tokens where mask == True
    return cross_entropy[mask].mean()


def compute_bleu(model, inp_lines: list[str] | np.ndarray[str], out_lines: list[str] | np.ndarray[str], bpe_sep: str = '@@ '):
    translations = [model.translate(line).replace(bpe_sep, '') for line in inp_lines]
    actual = [line.replace(bpe_sep, '') for line in out_lines]
    return corpus_bleu(
        list_of_references=[[ref.split()] for ref in actual],
        hypotheses=[trans.split() for trans in translations],
        smoothing_function=SmoothingFunction().method2,
    ) * 100


def compute_lr(d_model, step_num, warmup_steps=4000):
    return np.power(d_model, -0.5) * min(np.power(step_num, -0.5), step_num * np.power(warmup_steps, -1.5))


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def main():
    data_inp = np.array(open('train.bpe.ru').read().split('\n'))
    data_out = np.array(open('train.bpe.en').read().split('\n'))

    # numpy arrays
    train_inp, val_inp, train_out, val_out = train_test_split(data_inp, data_out, test_size=3000, random_state=42)

    inp_vocab = Vocab(train_inp)
    out_vocab = Vocab(train_out)

    batch_size = 4
    train_dataset = TranslationDataset(train_inp, train_out)

    def collate_fn(data):
        inp_texts, out_texts = zip(*data)
        inp_batch = inp_vocab.to_matrix(inp_texts).to(device)
        out_batch = out_vocab.to_matrix(out_texts).to(device)
        return inp_batch, out_batch

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    d_model = 16

    model = Transformer(
        inp_vocab=inp_vocab,
        out_vocab=out_vocab,
        d_model=d_model,
        d_k=16,
        d_v=16,
        d_ff=64,
        n_layers=6,
        n_heads=2,
        max_seq_len=110
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    n_epochs = 1
    val_dataloader = DataLoader(TranslationDataset(val_inp, val_out), batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn)
    current_step = 0
    warmup_steps = 4000
    val_steps = 100

    model.train()
    for epoch in range(n_epochs):
        for inp_batch, out_batch in tqdm(train_dataloader, desc=f'Epoch {epoch}'):
            # set lr
            lr = compute_lr(d_model, current_step + 1, warmup_steps)
            set_lr(opt, lr)

            # train
            loss = compute_loss(model, inp_batch, out_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            print("train_loss", loss, current_step)
            current_step += 1

            # validation
            if current_step % val_steps == 0:
                model.eval()
                val_loss = 0.
                batch_count = 0
                with torch.no_grad():
                    for val_inp_batch, val_out_batch in val_dataloader:
                        loss = compute_loss(model, val_inp_batch, val_out_batch)
                        val_loss += loss.item()
                        batch_count += 1
                print("val_loss", val_loss / batch_count, epoch)
                model.train()


if __name__ == '__main__':
    main()
