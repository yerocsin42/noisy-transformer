import torch
import torch.nn as nn
from torch.nn import functional as F
import random

"""
ALTERNATIVE_7: Perturbed Token Mixup
- Idea: Combine (or mix) embeddings of different tokens from the same sequence or from different sequences
        with some noise. This method is inspired by the mixup technique used in image processing, promoting
        smoother transitions between different token representations.

-Implementation: For each token in a sequence, mix its embedding with that of a random token from the same or
                 a different sequence, adding some noise during the combination.

origin of idea: https://medium.com/@lhungting/mixup-a-trivial-but-powerful-image-augmentation-technique-4e2d0725b8e3#:~:text=MixUp%20augmentation%20linearly%20combines%20an,sampled%20from%20a%20Beta%20distribution.



- Mixup Probability (mixup_prob): This parameter determines how often the Perturbed Token Mixup is applied during training.
                                  A higher probability means more frequent mixups.

- Lambda (lam): A random mixing coefficient that determines the ratio between the original and the permuted token embeddings.
              The closer lam is to 1, the more the mixed embedding resembles the original one.

- Noise Addition (noise_std): This parameter controls the standard deviation of the Gaussian noise added to the mixed embeddings.
                            The noise helps the model to handle variations in the input representations.

- Mixup Mechanism: If the model is in training mode (self.training) and a random value is less than mixup_prob, the token embeddings are mixed
                with those of another random batch. This mixup creates new training examples by combining inputs from different sequences,
                effectively augmenting the training data.

- Integration with Transformer Blocks: After applying the mixup and noise, the perturbed embeddings are passed through the transformer blocks as usual.

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

    def __init__(self, d_model, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # Each head gets a smaller dimensional representation of the data.
        d_head = d_model // n_head
        self.sa = MultiHeadAttention(n_head, d_head)
        self.ff = FeedFoward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        This is different from te originl transformer paper
        In the "Attention is all you Need" paper, we had
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffwd(x))
        See Figure 1 here, and mimic that: https://arxiv.org/pdf/2002.04745.pdf
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding layers
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(
            *[DecoderBlock(d_model, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, mixup_prob=0.2, noise_std=0.01):
        B, T = idx.shape

        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, d_model)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, d_model)
        x = tok_emb + pos_emb  # (B, T, d_model)

        # Apply Perturbed Token Mixup
        if self.training and random.random() < mixup_prob:
            # Select a random batch for mixup
            perm = torch.randperm(B).to(device)
            lam = torch.rand(1).item()  # Lambda for mixing
            x = lam * x + (1 - lam) * x[perm]
            # Add Gaussian noise for perturbation
            x += noise_std * torch.randn_like(x)

        # Pass through transformer blocks
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

def get_model5():
    return GPT()

