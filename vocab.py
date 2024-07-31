import numpy as np
import torch


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
        actual_lengths = [len(text) for text in tokenized_texts]
        # create a matrix of size (batch_size, max_len),
        # where max_len is the length of the longest sequence in the batch
        matrix = np.full((batch_size, max(actual_lengths)), fill_value=self.eos_idx)  # using EOS as padding token

        for i, tokens in enumerate(tokenized_texts):
            matrix[i, :actual_lengths[i]] = [self.token_to_index.get(token, self.unk_idx) for token in tokens]

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
