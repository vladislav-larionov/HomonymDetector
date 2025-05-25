import torch

print("PyTorch Version : {}".format(torch.__version__))


import torchtext

print("TorchText Version : {}".format(torchtext.__version__))

from torch.utils.data import DataLoader

train_dataset, test_dataset  = torchtext.datasets.AG_NEWS()
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")

def build_vocabulary(datasets):
    for dataset in datasets:
        for _, text in dataset:
            yield tokenizer(text)

vocab = build_vocab_from_iterator(build_vocabulary([train_dataset, test_dataset]), min_freq=1, specials=["<UNK>"])
vocab.set_default_index(vocab["<UNK>"])
tokens = tokenizer("Hello how are you?, Welcome to CoderzColumn!!")
indexes = vocab(tokens)

print(tokens, indexes)