import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)

def build_vocab(train_file):
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(yield_tokens(load_data(train_file), tokenizer), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

class TextDataset(Dataset):
    def __init__(self, data_iter, vocab, tokenizer, seq_len=32):
        self.data = []
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        for text in data_iter:
            tokens = [vocab["<bos>"]] + [vocab[token] for token in tokenizer(text)] + [vocab["<eos>"]]
            for i in range(0, len(tokens) - seq_len):
                self.data.append((tokens[i:i+seq_len], tokens[i+1:i+seq_len+1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])