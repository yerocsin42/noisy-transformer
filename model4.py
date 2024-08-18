import torch
import torch.nn as nn
from torch.nn import functional as F
import random


eval_iters = 200
d_model = 96 # could increase this to 386 --> in order to make the model bigger.
n_head = 6 # This implied that each head has a dimension for the key, query, and values of d_model / 6.
n_layer = 6 # This implies we have 6 turns to mix the embeddigs --- `n_layer` is "Nx" in the paper.
dropout = 0.2
vocab_size = 62
batch_size = 64 # can infer the # of independent sequences we will process in parallel from here.
block_size = 256 # can infer the maximum context length for predictions from here.
max_iters = 5000
eval_interval = 500 # answers how often we evaluate across the optimization: every 500 iterations
learning_rate = 3e-4 # can set to different values
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
ALTENATIVE_10: DropConnect in Transformer Weights
- Idea: Instead of dropping out entire neurons or units, randomly drop out individual connections (weights)
within the transformer layers. This is known as DropConnect and can lead to sparser, more efficient learning.
- Implementation: Apply dropout to the weights of the transformer layers (not the activations), randomly setting a fraction of the weights to zero during training.

- DropConnect Function: This function applies DropConnect by randomly setting a portion of the weights to zero based on the drop_prob.

- Modifications in Head and MultiHeadAttention: DropConnect is applied to the weights of the linear layers in the Head and MultiHeadAttention
  classes. This regularizes the attention mechanism by zeroing out random weights.

- Modifications in FeedForward: Similarly, DropConnect is applied to the weights in the feed-forward network, adding regularization to the transformation layers.

- drop_prob: This parameter controls the probability of each weight being set to zero. You can tune this parameter based on your experiment.-

"""

def dropconnect(layer, drop_prob):
    """Apply DropConnect to the weights of the given layer."""
    if not layer.training or drop_prob == 0:
        return layer.weight
    # Create a binary mask with the same shape as the weights
    mask = torch.bernoulli(torch.ones_like(layer.weight) * (1 - drop_prob))
    # Apply the mask to the weights
    return layer.weight * mask

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, d_head, drop_prob=0.1):
        super().__init__()
        self.d_head = d_head
        self.drop_prob = drop_prob
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, d = x.shape
        # Apply DropConnect to the linear layers' weights
        k = F.linear(x, dropconnect(self.W_K, self.drop_prob))   # (B, T, d_head)
        q = F.linear(x, dropconnect(self.W_Q, self.drop_prob))   # (B, T, d_head)
        v = F.linear(x, dropconnect(self.W_V, self.drop_prob))   # (B, T, d_head)

        scores = q @ k.transpose(-2, -1) * self.d_head ** -0.5  # (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        a = F.softmax(scores, dim=-1)  # (B, T, T)
        a = self.dropout(a)
        out = a @ v  # (B, T, d_head)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, d_head, drop_prob=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_head, drop_prob) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.drop_prob = drop_prob

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply DropConnect to the projection layer's weights
        out = F.linear(out, dropconnect(self.proj, self.drop_prob))
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    """
    A simple linear layer followed by a non-linearity; this is applied at the token level.
    """

    def __init__(self, d_model, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        d_ff = 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply DropConnect to the weights of the linear layers
        x = F.linear(x, dropconnect(self.fc1, self.drop_prob))
        x = F.relu(x)
        x = F.linear(x, dropconnect(self.fc2, self.drop_prob))
        return self.dropout(x)

class DecoderBlock(nn.Module):
    """
    Transformer decoder block: communication followed by computation.
    These are stacked on top of each other one after another.
    """

    def __init__(self, d_model, n_head, drop_prob=0.1):
        super().__init__()
        d_head = d_model // n_head
        self.sa = MultiHeadAttention(n_head, d_head, drop_prob)
        self.ff = FeedFoward(d_model, drop_prob)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(
            *[DecoderBlock(d_model, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.ff(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx


def get_model4():
    return GPT()

