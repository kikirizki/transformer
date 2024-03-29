import os
import torch
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader

from dataset import Multi30kDatasetEN_DE
from vanilla_transformer import make_model
from pathlib import Path
from tqdm import tqdm

d_model = 512
n_heads = 64
batch_Size = 3
ff_hidden_size = 2048

dropout_prob = 0.1
save_interval = 1
num_epochs = 100

checkpoint_path = "checkpoint"
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

en_de_dataset = Multi30kDatasetEN_DE()
dataset_loader = DataLoader(en_de_dataset, batch_size=batch_Size, drop_last=True)
n_german_vocabs = len(en_de_dataset.german_vocab)
n_english_vocabs = len(en_de_dataset.english_vocab)
german_max_seq = en_de_dataset.german_max_seq
english_max_seq = en_de_dataset.english_max_seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = VanillaTransformer(d_model, n_german_vocabs, ff_hidden_size, n_heads, dropout_prob).to(device)
model = make_model(n_english_vocabs,n_german_vocabs).to(device)
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss()

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    return mask


def train(model, dataset_loader, optimizer, criterion, num_epochs):
    tgt_mask = generate_square_subsequent_mask(62).unsqueeze(-1).unsqueeze(-1)
    src_mask = torch.ones((62,62), device=device).type(torch.bool).unsqueeze(-1).unsqueeze(-1)

    model.train()
    for epoch in range(num_epochs):
        loss_epoch = 0.0
        for batch in tqdm(dataset_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            encoder_input = rearrange(x, "batch_size sequence_length -> sequence_length batch_size")
            y = rearrange(y, "batch_size sequence_length -> sequence_length batch_size")

            decoder_input = y[:-1, :]

            output = model(encoder_input, decoder_input, src_mask, tgt_mask)

            target = torch.nn.functional.one_hot(y % 1, num_classes=n_german_vocabs)[1:, :].view(-1,
                                                                                                 n_german_vocabs).to(
                float)

            loss = criterion(output.view(-1, n_german_vocabs), target)

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
    train(model, dataset_loader, optimizer, criterion, num_epochs)

