import math
import torch
import torch.nn as nn
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model%num_heads == 0
        self.d_k = d_model//num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.soft_argmax = nn.Softmax(-1)
        self.split_head = nn.Linear(d_model, self.d_k * num_heads)
        self.W_o = nn.Linear(self.num_heads * self.d_k, d_model)

    def split(self, x):
        x = self.split_head(x)
        x = rearrange(x, "seq_length batch_size (heads d_k) -> seq_length batch_size heads d_k", heads=self.num_heads)
        return x

    def forward(self, query, key, value, mask=None):
        query, key, value = self.split(query), self.split(key), self.split(value)
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


class TransformersLayer(nn.Module):
    def __init__(self,
                 d_model,
                 self_attention: MultiHeadAttention,
                 feed_forward: FeedForward,
                 dropout_probs: float,
                 source_attention: MultiHeadAttention = None,
                 source_mask=None):
        super(TransformersLayer, self).__init__()
        self.self_attention = self_attention
        self.source_attention = source_attention
        self.source_mask = source_mask
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_probs)

        self.layernorm_self_attention = nn.LayerNorm([d_model])
        self.layernorm_feedforward = nn.LayerNorm([d_model])
        if source_attention is not None:
            self.layernorm_source_attention = nn.LayerNorm([d_model])

    def forward(self, x, src=None, src_mask=None):
        x_ = self.layernorm_self_attention(x)
        self_attention_matrices = self.self_attention(key=x_, query=x_, value=x_)
        x = self.dropout(self_attention_matrices) + x

        if src is not None:
            x_ = self.layernorm_source_attention(x)
            source_attention_matrices = self.source_attention(query=x_, key=src, value=src)
            x = self.dropout(source_attention_matrices) + x

        x_ = self.layernorm_feedforward(x)
        ff = self.feed_forward(x_)
        x = self.dropout(ff) + x
        return x


class TransformersEncoder(nn.Module):
    def __init__(self, d_model, ff_hidden_size, n_heads, dropout_prob):
        super(TransformersEncoder, self).__init__()
        self.feed_forward = FeedForward(d_model, ff_hidden_size, dropout_prob)
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.transformer_layer = TransformersLayer(d_model, self.self_attention, self.feed_forward, dropout_prob)

    def forward(self, x):
        return self.transformer_layer(x)


class TransformersDecoder(nn.Module):
    def __init__(self, d_model, ff_hidden_size, n_heads, dropout_prob):
        super(TransformersDecoder, self).__init__()
        self.feed_forward = FeedForward(d_model, ff_hidden_size, dropout_prob)
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.source_attention = MultiHeadAttention(d_model, n_heads)
        self.transformer_layer = TransformersLayer(d_model,
                                                   self.self_attention,
                                                   self.feed_forward,
                                                   dropout_prob,
                                                   self.source_attention)

    def forward(self, x, src):
        return self.transformer_layer(x, src)


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def forward(self,x):
        return x