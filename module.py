import torch
import torch.nn as nn
import copy
import math
from einops import rearrange
from torch.nn.functional import log_softmax


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embd, tgt_embd, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embd = src_embd
        self.tgt_embd = tgt_embd
        self.generator = generator

    def forward(self, x, tgt, mask, src_mask):
        return self.generator(self.decode(tgt, self.encode(x, src_mask), mask, src_mask))

    def encode(self, x, mask):
        return self.encoder(self.src_embd(x), mask)

    def decode(self, x, memory, mask, src_mask):
        return self.decoder(self.tgt_embd(x), memory, mask, src_mask)


def clones(layer, n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])


class Encoder(nn.Module):
    def __init__(self, d_model, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm([d_model])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout_prob):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm([d_model])
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, sublayer):
        return self.dropout(self.norm(sublayer(x))) + x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.skip_conn = clones(AddNorm(d_model, dropout), 2)
        self.self_attn = self_attn
        self.ffn = feed_forward

    def forward(self, x, mask):
        x = self.skip_conn[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.skip_conn[0](x, self.ffn)


class Decoder(nn.Module):
    def __init__(self, d_model, layer, n):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm([d_model])

    def forward(self, x, mem, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, mem, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout_prob):
        super(DecoderLayer, self).__init__()
        self.skip_conn = clones(AddNorm(d_model, dropout_prob), 3)
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ffn = feed_forward

    def forward(self, x, mem, src_mask, tgt_mask):
        x = self.skip_conn[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.skip_conn[0](x, lambda x: self.self_attn(x, mem, mem, tgt_mask, src_mask))
        return self.skip_conn[0](x, self.ffn)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.soft_argmax = nn.Softmax(1)
        self.query_proj = nn.Linear(d_model, self.d_k * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_k * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_k * num_heads)
        self.W_o = nn.Linear(self.num_heads * self.d_k, d_model)

    def split_head(self, x):
        x = rearrange(x, "seq_length batch_size (heads d_k) -> seq_length batch_size heads d_k", heads=self.num_heads)
        return x

    def forward(self, query, key, value, mask=None, src_mask=None):
        query, key, value = self.query_proj(query), self.key_proj(key), self.value_proj(value)
        query, key, value = self.split_head(query), self.split_head(key), self.split_head(value)
        score = torch.einsum(
            "q b h d, k b h d -> q k b h", query, key)
        score /= math.sqrt(self.d_k)
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        attention_matrix = self.soft_argmax(score)
        out = torch.einsum("q k b h, k b h d-> q b h d", attention_matrix, value)
        out = rearrange(out, "seq_length batch_size heads d_k -> seq_length batch_size (heads d_k)",
                        heads=self.num_heads)
        out = self.W_o(out)
        return out


class EmbeddingWithLearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, n_vocab, max_sequence=5000):
        super(EmbeddingWithLearnablePositionalEncoding, self).__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros([max_sequence, 1, d_model]), requires_grad=True)
        self.d_model = d_model

    def forward(self, sparse_input):
        pe = self.positional_encoding[:sparse_input.shape[0]]
        return self.linear(sparse_input) * math.sqrt(self.d_model) + pe


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_size, dropout_prob):
        super(FeedForward, self).__init__()
        self.layer_list = nn.ModuleList([
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, d_model)
        ])

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x
