import torch
import torch.nn as nn
from torch.nn import functional as F
import random

"""
ALTERNATIVE_9: Noisy Residual Connections

- Idea: Add noise to the residual connections that skip layers. This can make the learning process more resilient to errors in the intermediate layers
and encourage the model to learn more robust representations.
- Implementation: Inject Gaussian or uniform noise into the residual connection before adding it back to the layer’s output.

- noise_std: This parameter controls the amount of noise added to the residual connections. You can adjust it based on your experiment.
- torch.randn_like(x): Generates a tensor of the same shape as x with values drawn from a standard normal distribution.
- Adding Noise: Noise is added after the residual connection is computed, making the output slightly perturbed.
This noise is added twice: once after the self-attention block and once after the feed-forward block.

"""

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

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, d_head):
        super().__init__()
        self.d_head  = d_head
        # Map each key, query, or value in to a d_head dimensional model.
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, T, d_model)
        B,T,d = x.shape
        k = self.W_K(x)   # (B,T,d_head)
        q = self.W_Q(x) # (B,T,head_size)
        # compute attention scores ("affinities")

        # (B T, d) @ (B, d, T) = (B, T, T)
        scores = q @ k.transpose(-2,-1) * self.d_head**-0.5 # (B, T, d_head) @ (B, d_head, T) -> (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        a = F.softmax(scores, dim=-1) # (B, T, T)
        a = self.dropout(a)
        # perform the weighted aggregation of the values
        v = self.W_V(x) # (B,T,d)
        out = a @ v # (B, T, T) @ (B, T, d) -> (B, T, d)
        # These are the values.
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, d_head):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_head) for _ in range(num_heads)])
        # This is to project back to the dimension of d_model. In this case, it is just a learned linear map.
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the different representations per head.
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project the concatenation.
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """
    A simple linear layer followed by a non-linearity; this is applied at the token level.
    """

    def __init__(self, d_model):
        super().__init__()
        d_ff = 4 * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.ff(x)


class DecoderBlock(nn.Module):
    """
    Transformer decoder block: communication followed by computation.
    These are stacked on top of each other one after another.
    """

    def __init__(self, d_model, n_head, noise_std=0.1):
        super().__init__()
        d_head = d_model // n_head
        self.sa = MultiHeadAttention(n_head, d_head)
        self.ff = FeedFoward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.noise_std = noise_std  # Standard deviation for the Gaussian noise

    def forward(self, x):
        # Apply self-attention and add Gaussian noise to the residual connection
        residual = x
        x = self.ln1(x)
        x = self.sa(x)
        x = residual + x + torch.randn_like(x) * self.noise_std

        # Apply feed-forward network and add Gaussian noise to the residual connection
        residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = residual + x + torch.randn_like(x) * self.noise_std

        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(
            *[DecoderBlock(d_model, n_head=n_head) for _ in range(n_layer)]
        )
         # final layer norm
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # (B,T,d_model)
        tok_emb = self.token_embedding_table(idx)
        # (T,d_model)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # Add positional encodings.
        # (B,T,C)
        x = tok_emb + pos_emb

        # Mix up the token representations over and over via the blocks
        # (B,T,C)
        x = self.blocks(x)
        # (B,T,C)
        x = self.ln(x)
        # (B,T,vocab_size)
        logits = self.ff(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        idx is (B, T) array of indices in the current context.
        This will generate B total paths in parrallel.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            # The model only has kowledge of the context of maximum size block_size.
            idx_cond = idx[:, -block_size:]
            # get the predictions
            # (B, T, vocab_size)
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        self.train()
        return idx


def get_model3():
    return GPT()

