import copy
import torch.nn as nn
from module import EmbeddingWithLearnablePositionalEncoding, \
    EncoderLayer, \
    DecoderLayer, \
    Encoder, \
    Decoder, \
    FeedForward, \
    Generator, \
    MultiHeadAttention, \
    EncoderDecoder


def make_model(
        src_vocab, tgt_vocab, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadAttention(d_model, h)
    ff = FeedForward(d_model, d_ff, dropout)
    src_embd = EmbeddingWithLearnablePositionalEncoding(d_model, src_vocab)
    tgt_embd = EmbeddingWithLearnablePositionalEncoding(d_model, tgt_vocab)
    model = EncoderDecoder(
        Encoder(d_model, EncoderLayer(d_model, c(attn), c(ff), dropout), n),
        Decoder(d_model, DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n),
        src_embd,
        tgt_embd,
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
