import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import AliceInTheWonderlandDataset
import numpy as np
from model import Transformer
from einops import rearrange

d_model = 512
n_heads = 2
batch_Size = 12
ff_hidden_size = 2
n_words = 7
dropout_prob = 0.8

alice_dataset = AliceInTheWonderlandDataset(n_words)
dataset_loader = DataLoader(alice_dataset, batch_size=batch_Size, drop_last=True)
n_vocabs = alice_dataset.get_vocabulary_length()

model = Transformer(d_model, n_vocabs, ff_hidden_size, n_heads, dropout_prob)
optimizer = torch.optim.Adam(params=model.parameters())
criterion = nn.CrossEntropyLoss()


def train(model, dataset_loader, optimizer, criterion):
    model.train()
    for i, batch in enumerate(dataset_loader):
        x, y = batch
        encoder_input = rearrange(x, "batch_size sequence_length -> sequence_length batch_size")
        y = rearrange(y, "batch_size sequence_length -> sequence_length batch_size")

        decoder_input = y[:-1, :]

        output = model(encoder_input, decoder_input)
        target = torch.nn.functional.one_hot(y % 1, num_classes=n_vocabs)[1:, :].view(-1, n_vocabs).to(float)

        loss = criterion(output.view(-1, n_vocabs), target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()


if __name__ == '__main__':
    train(model, dataset_loader, optimizer, criterion)
