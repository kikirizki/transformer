import torch
import torch.nn as nn
from module import EmbeddingWithLearnablePositionalEncoding, TransformersEncoder, TransformersDecoder


class VanillaTransformer(nn.Module):
    def __init__(self, d_model, n_vocab, ff_hidden_size, n_heads, dropout_prob):
        super(VanillaTransformer, self).__init__()
        self.embedding = EmbeddingWithLearnablePositionalEncoding(d_model, n_vocab)
        self.encoder = TransformersEncoder(d_model, ff_hidden_size, n_heads, dropout_prob)
        self.decoder = TransformersDecoder(d_model, ff_hidden_size, n_heads, dropout_prob)
        self.linear = torch.nn.Linear(d_model, n_vocab)


    def forward(self, encoder_input, decoder_input, target_mask=None, source_mask=None):
        encoder_output = self.encoder(self.embedding(encoder_input), source_mask)
        decoder_output = self.decoder(self.embedding(decoder_input), src=encoder_output, mask=target_mask)
        output = self.linear(decoder_output)
        return output
