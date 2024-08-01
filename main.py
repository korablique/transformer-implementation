import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import WordPunctTokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split
from subword_nmt.apply_bpe import BPE
from subword_nmt.learn_bpe import learn_bpe
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformer_implementation.transformer import Transformer
from transformer_implementation.vocab import Vocab

device = 'cuda'

tokenizer = WordPunctTokenizer()


def tokenize(x: str) -> str:
    """
    :param x: text
    :return: tokenized text, tokens are separated by whitespace
    """
    return ' '.join(tokenizer.tokenize(x.lower()))


class TranslationDataset(Dataset):
    def __init__(self, inp_texts: np.ndarray[str], out_texts: np.ndarray[str]):
        assert len(inp_texts) == len(out_texts), 'inp_texts and out_texts must be the same size'
        self.inp_texts = inp_texts
        self.out_texts = out_texts

    def __len__(self) -> int:
        return len(self.inp_texts)

    def __getitem__(self, idx: int):
        return self.inp_texts[idx], self.out_texts[idx]  # input string and output string


def compute_loss(model: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """
    :param model:
    :param inp: input tokens matrix
    :param out: reference tokens matrix
    """
    logits = model(inp, out)  # (batch_size, out_len, out_vocab_size)
    return F.cross_entropy(
        logits.contiguous().view(-1, len(model.out_vocab)),
        out.contiguous().view(-1),
        ignore_index=model.out_vocab.eos_idx)


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
    train_data_folder = './train_data'
    data_path = os.path.join(train_data_folder, 'data.txt')

    # split data.txt onto two files - en и ru
    with open(os.path.join(train_data_folder, 'train.en'), 'w') as f_en, \
            open(os.path.join(train_data_folder, 'train.ru'), 'w') as f_ru:
        for line in open(data_path):
            line_en, line_ru = line.strip().split('\t')
            f_en.write(tokenize(line_en) + '\n')
            f_ru.write(tokenize(line_ru) + '\n')

    # learn BPE
    bpe = {}
    for lang in ['en', 'ru']:
        learn_bpe(
            open(os.path.join(train_data_folder, 'train.' + lang)),
            open(os.path.join(train_data_folder, 'bpe_rules.' + lang), 'w'),
            num_symbols=8000
        )
        bpe[lang] = BPE(open(os.path.join(train_data_folder, 'bpe_rules.' + lang)))
        # save BPE rules
        with open(os.path.join(train_data_folder, 'train.bpe.' + lang), 'w') as f_out:
            for line in open(os.path.join(train_data_folder, 'train.' + lang)):
                # apply BPE to train data
                f_out.write(bpe[lang].process_line(line.strip()) + '\n')

    # read train data
    data_inp = np.array(open('train_data/train.bpe.ru').read().split('\n'))
    data_out = np.array(open('train_data/train.bpe.en').read().split('\n'))

    train_inp, val_inp, train_out, val_out = train_test_split(data_inp, data_out, test_size=3000, random_state=42)

    inp_vocab = Vocab(train_inp)
    out_vocab = Vocab(train_out)

    train_dataset = TranslationDataset(train_inp, train_out)

    def collate_fn(data):
        inp_texts, out_texts = zip(*data)
        inp_batch = inp_vocab.to_matrix(inp_texts).to(device)
        out_batch = out_vocab.to_matrix(out_texts).to(device)[:, 1:]
        return inp_batch, out_batch

    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    d_model = 256

    model = Transformer(
        inp_vocab=inp_vocab,
        out_vocab=out_vocab,
        d_model=d_model,
        d_k=64,
        d_v=64,
        d_ff=512,
        n_layers=6,
        n_heads=3,
        max_seq_len=150
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    n_epochs = 100
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
