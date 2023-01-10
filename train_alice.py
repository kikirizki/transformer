import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import AliceInTheWonderlandDataset

from model import Transformer
from einops import rearrange

d_model = 512
n_heads = 32
batch_Size = 12
n_words = 7
dropout_prob = 0.8

alice_dataset = AliceInTheWonderlandDataset(n_words)
dataset_loader = DataLoader(alice_dataset, batch_size=batch_Size, drop_last=True)
n_vocabs = alice_dataset.get_vocabulary_length()

model = Transformer(d_model, n_vocabs)
optimizer = torch.optim.Adam(params=model.parameters())
criterion = nn.CrossEntropyLoss()

def train(model, dataset_loader, optimizer, criterion):
    model.train()
    for i, batch in enumerate(dataset_loader):
        x, y = batch
        x = rearrange(x,"batch_size sequence_length -> sequence_length batch_size")
        y = rearrange(y,"batch_size sequence_length -> sequence_length batch_size")

        print(x.shape)
        out = model(x)
        print(out.shape)
        break


if __name__ == '__main__':
    train(model, dataset_loader, None, criterion)
