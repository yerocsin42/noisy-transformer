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
from model1 import get_model1
from model2 import get_model2
from model3 import get_model3
from model4 import get_model4
from model5 import get_model5
import argparse
import sys

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
        5: get_model5}
    
    model_func = model_dict.get(model_choice)
    
    if model_func is None:
        raise ValueError("Invalid model choice")
    
    return model_func()


def train(model_choice):

    train_acc = []
    val_acc = []

    #model = model_choice().to(device)
    
    model = get_model5().to(device)

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

    with open(f'accuracies', mode='a', newline='') as file:
        writer = csv.writer(file)
        for epoch, (train_acc, val_acc) in enumerate(zip(train_acc, val_acc)):
            writer.writerow([epoch + 1, train_acc, val_acc])


#def decode():
    # Start the model with a new line, generate up to 10000 tokens
    # This is technically doing generations in batches, but here we have a batch size of 1 and 1 element to start in the batch
 #   context = torch.zeros((1, 1), dtype=torch.long, device=device)
  #  print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
   # open('fake_hemingway.txt', 'w').write(decode(model.generate(context, max_new_tokens=100)[0].tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("param",type=int,nargs='?',help="Model Number – 1=Baseline, 2=GN Inject, 3=ResidualConn, 4=DropConn, 5=PerturbedTokens")
    args = parser.parse_args()
    
    if args.param is None:
        print("Error: No parameter provided.")
        parser.print_help()
        sys.exit(1)

    if args.param > 5 or args.param < 1:
        raise ValueError("Error: Model Number needs to be between 1-5.")

    model_no = load_model(args.param)

    train(model_no)








