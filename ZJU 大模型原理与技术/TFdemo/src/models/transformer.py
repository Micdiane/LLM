import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N = x.shape[0]
        length = x.shape[1]

        # Split the embedding into multiple heads
        values = self.values(x).view(N, length, self.heads, self.head_dim)
        keys = self.keys(x).view(N, length, self.heads, self.head_dim)
        queries = self.queries(x).view(N, length, self.heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)  # (N, heads, length, head_dim)
        keys = keys.permute(0, 2, 1, 3)      # (N, heads, length, head_dim)
        queries = queries.permute(0, 2, 1, 3)  # (N, heads, length, head_dim)

        energy = torch.einsum("nqhd,nkhd->nqk", [queries, keys])  # (N, heads, query_length, key_length)
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("nqk,nvhd->nqhd", [attention, values]).reshape(
            N, length, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, hidden_size, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        x = self.dropout(self.norm2(forward + x))
        return x

class Transformer(nn.Module):
    def __init__(self, embed_size, heads, hidden_size, num_layers, dropout, input_dim):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, input_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)