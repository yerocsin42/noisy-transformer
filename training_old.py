#!/usr/bin/env python
# coding: utf-8



"""
We first implement a GPT1-like language model to generate text. The training data looks like (y0, y1, y2) -> (y1, y2, y3) and we have 3 loss terms. The model will be trained on chunks of data from Hemingway's most well-known novel. The default setting below will produce a model with about 700K parameters. The model should be expected to work better and better as we make the sizes of the parameters (starting with d_model below) bigger and bigger. Collab should allow us to scale to a 10M parameter model without making us to pay.

At the end, after training, you'll decode with this model and generate text. Although there are specialized metric for this task (e.g. perplexity), we will simplify the comparisons by just considering training and validation accuracies.

The goal of this notebook is to incorporate from various angles noise structures that hypothetically would promise something interesting or surprising.

"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import csv

###################################################################################################################

"""
BASELINE

"""

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
    

def get_model():
    return GPT()

############################################################################################



# hyperparameters
batch_size = 64 # can infer the # of independent sequences we will process in parallel from here.
block_size = 256 # can infer the maximum context length for predictions from here.
max_iters = 5000
eval_interval = 500 # answers how often we evaluate across the optimization: every 500 iterations
learning_rate = 3e-4 # can set to different values
"""
Use 'mps' if on a mac as below:

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# How many batches we use each time we evaluate
eval_iters = 200
d_model = 96 # could increase this to 386 --> in order to make the model bigger.
n_head = 6 # This implied that each head has a dimension for the key, query, and values of d_model / 6.
n_layer = 6 # This implies we have 6 turns to mix the embeddigs --- `n_layer` is "Nx" in the paper.
dropout = 0.2
# ------------

torch.manual_seed(1337)


with open('hemingway.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) / train_loss > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def load_model(model_choice):
    model_dict = {
        1: get_model1,
        2: get_model2,
        3: get_model3,
        4: get_model4,
        5: get_model5
    }
    
    model_func = model_dict.get(model_choice)
    
    if model_func is None:
        raise ValueError("Invalid model choice")
    
    return model_func()


def train():#model_choice):

    train_acc = []
    val_acc = []

    model = GPT().to(device)
    # Print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
#    print(f"Model Number: {model_choice}")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    early_stopping = EarlyStopping(tolerance=1, min_delta=0.2)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % 50 == 0:
            losses = estimate_loss(model)
            train_acc.append(losses['train'])
            val_acc.append(losses['val'])
        
        if iter >=3500:
            print(f"We stop at step {iter}")
            break

        if iter % eval_interval == 0 or iter == max_iters - 1:
            if iter:
                scheduler.step()
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            early_stopping(losses['train'], losses['val'])
            if early_stopping.early_stop:
                print("We stop at epoch {}".format(iter))
                break


        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'gpt.pt')

    with open('accuracies', mode='a', newline='') as file:
        writer = csv.writer(file)
        for epoch, (train_acc, val_acc) in enumerate(zip(train_acc, val_acc)):
            writer.writerow([epoch + 1, train_acc, val_acc])


def decode():
    # Start the model with a new line, generate up to 10000 tokens
    # This is technically doing generations in batches, but here we have a batch size of 1 and 1 element to start in the batch
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
    open('fake_hemingway.txt', 'w').write(decode(model.generate(context, max_new_tokens=100)[0].tolist()))


if __name__ == "__main__":
    #model_choice = int(input("Model number (1-5): "))
    train()#model_choice)







