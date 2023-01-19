import os
import torch
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader

from dataset import AliceInTheWonderlandDataset, Multi30kDatasetEN_DE
from vanilla_transformer import VanillaTransformer
from pathlib import Path

d_model = 512
n_heads = 2
batch_Size = 12
ff_hidden_size = 2
n_words = 7
dropout_prob = 0.1
save_interval = 5
num_epochs = 100
checkpoint_path = "checkpoint"
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)


alice_dataset = AliceInTheWonderlandDataset(n_words)
dataset_loader = DataLoader(alice_dataset, batch_size=batch_Size, drop_last=True)
n_vocabs = alice_dataset.get_vocabulary_length()


model = VanillaTransformer(d_model, n_vocabs, ff_hidden_size, n_heads, dropout_prob)
optimizer = torch.optim.Adam(params=model.parameters())
criterion = nn.CrossEntropyLoss()

def train(model, dataset_loader, optimizer, criterion, num_epochs):
    mask = torch.triu(torch.ones(n_words, n_words) * float('-inf'), diagonal=1)
    model.train()
    for epoch in range(num_epochs):
        loss_epoch = 0.0
        for i, batch in enumerate(dataset_loader):
            x, y = batch
            encoder_input = rearrange(x, "batch_size sequence_length -> sequence_length batch_size")
            y = rearrange(y, "batch_size sequence_length -> sequence_length batch_size")

            decoder_input = y[:-1, :]

            output = model(encoder_input, decoder_input, mask)

            target = torch.nn.functional.one_hot(y % 1, num_classes=n_vocabs)[1:, :].view(-1, n_vocabs).to(float)

            loss = criterion(output.view(-1, n_vocabs), target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            loss_epoch += loss.item()
        loss_epoch /= len(dataset_loader)
        if epoch % save_interval == 0:
          torch.save(model.state_dict(), os.path.join(checkpoint_path, f"checkpoint_{epoch}.pth"))
        print(f"Epoch {epoch} loss : {loss_epoch}")

if __name__ == '__main__':
    # train(model, dataset_loader, optimizer, criterion, num_epochs)
    dataset = Multi30kDatasetEN_DE()
    print(dataset[5])

