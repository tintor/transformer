import py7zr
import math
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import csv
import sys
import torch
import io
import random

from model import ModelArgs, Transformer


def load_compressed_csv(name: str) -> List[str]:
    print(f"Loading {name}")
    assert name.endswith('.7z')
    data = []
    with py7zr.SevenZipFile(name, mode='r') as archive:
        csv_bytes_io = archive.read()[name[:-3]]
        with io.TextIOWrapper(csv_bytes_io, encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            j = 0
            for row in reader:
                data.append(row['text'])
                if (j+1) % 100000 == 0:
                    print(f"{(j+1)//1000}k")
                j += 1
    return data


def build_vocabulary(data: List[str]) -> List[str]:
    vocab = set()
    for line in data:
        vocab |= set(line)
    vocab = sorted(list(vocab))
    vocab.extend(['<|end|>'])
    return vocab


device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

validation_data = load_compressed_csv('validation.csv.7z')
#train_data = load_compressed_csv('train.csv.7z')
train_data = validation_data

print("Building vocabulary")
vocab = build_vocabulary(train_data)

print(f"Lenth of train dataset in characters: {sum(len(s) for s in train_data)}")
print(f"Lenth of validation dataset in characters: {sum(len(s) for s in validation_data)}")
print(f"The vocabulary looks like this: {''.join(vocab)}\n")
print(f"Vocabulary size: {len(vocab)}")
assert ModelArgs.vocab_size == len(vocab), f"{ModelArgs.vocab_size} vs {len(vocab)}"

# Create a mapping between characters with corresponding integer indexes in vocabulary.
itos = vocab
stoi = {ch:i for i, ch in enumerate(vocab)}
# Tokenizer's encode function: take a string, output a list of integers
def encode_tokens(s: str) -> List[int]:
    return [stoi[ch] for ch in s]
# Tokenizer's decode function: take a list of integers, output a string
def decode_tokens(l: List[int]) -> str:
    return ''.join(itos[i] for i in l)

token_end = stoi['<|end|>']

prompt = "Hello World"
assert decode_tokens(encode_tokens(prompt)) == prompt

print(f"Example encoded tokens: {encode_tokens(prompt)}")

dataset = torch.tensor(encode_tokens(train_data[0]), dtype=torch.int).to(ModelArgs.device)
print(f"dataset-shape: {dataset.shape}")

def prepare_dataset(data: List[str], ) -> torch.Tensor:
    size = sum(len(s) + 1 for s in data)
    tokens = []
    for i, s in enumerate(data):
        tokens.extend(stoi[c] for c in s)
        tokens.append(token_end)
        if (i+1) % 100000 == 0:
            print(f"{(i+1)//1000}k / {len(data)//1000}k")
    tokens = torch.tensor(tokens, dtype=torch.uint8)
    return tokens


print("Prepare dataset")
validation_data = prepare_dataset(validation_data)
train_data = prepare_dataset(train_data)


def empty_dataset_batch(args: ModelArgs) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.full([args.max_batch_size, args.max_seq_len], token_end, dtype=torch.long).to(args.device)
    y = torch.full([args.max_batch_size], token_end, dtype=torch.long).to(args.device)
    return x, y


# Define function to generate batches from the given dataset
def get_random_dataset_batch(data: torch.Tensor, args: ModelArgs, x: torch.Tensor, y: torch.Tensor):
    x[:, :] = token_end
    for i in range(args.max_batch_size):
        e = random.randint(0, data.size(0)-1)
        s = max(0, e-args.max_seq_len)
        indices = torch.nonzero(data[s:e] == token_end)
        if indices.size(0) > 0:
            m = indices[-1]
            assert data[s + m] == token_end
            s = s + m + 1
        x[i, 0:e-s] = data[s:e]
        y[i] = data[e]


# Define a evaluate loss function to calculate and store training and validation loss for logging and plotting
@torch.no_grad()
def evaluate_loss(model: Transformer, args: ModelArgs) -> dict[str, float]:
    out = {}
    model.eval()

    x, y = empty_dataset_batch(args)

    losses = []
    for _ in range(10):      
        get_random_dataset_batch(train_data, args, x, y)
        _, loss = model(x=x, targets=y)
        losses.append(loss.item())
    out['train'] = np.mean(losses)
    
    losses = []
    for _ in range(10):      
        get_random_dataset_batch(validation_data, args, x, y)
        _, loss = model(x=x, targets=y)
        losses.append(loss.item())
    out['validation'] = np.mean(losses)

    model.train()
    return out


# Define a training function to perform model training
def train(model: Transformer, optimizer, args: ModelArgs) -> None:
    epochs = args.epochs
    log_interval = args.log_interval
    device = args.device
    losses = []   
    start_time = time.time()

    x, y = empty_dataset_batch(args)
    for epoch in range(epochs):
        optimizer.zero_grad()

        get_random_dataset_batch(train_data, args, x, y)
        logits, loss = model(x=x.to(device), targets=y.to(device))
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model, args)
            losses += [x]
            percent = (epoch + 1) * args.max_batch_size / train_data.size(0) * 100
            print(f"Epoch {epoch} | train loss {x['train']:.3f} | val loss {x['validation']:.3f} | Time {batch_time:.3f} | Dataset {percent:.6f}%")
            start_time = time.time()

        if epoch % 50 == 0:
            generate(model, "Once upon a time there was a cat named Elle", ModelArgs, max_gen_len=100)
    
    torch.save({'args': args.__dict__, 'state_dict': model.state_dict()}, "transformer.pth")


def generate(model: Transformer, prompt: str, params: ModelArgs, max_gen_len: int=500, temperature: float = 0.6, top_p: float = 0.9) -> None:
    # prompt_tokens: List of user input texts or prompts
    # max_gen_len: Maximum length of the generated text sequence.
    # temperature: Temperature value for controlling randomness in sampling. Defaults to 0.6.
    # top_p: Top-p probability threshold for sampling prob output from the logits. Defaults to 0.9.
    batch_size = 1
    prompt_tokens = encode_tokens(prompt)
    assert len(prompt_tokens) <= params.max_seq_len
    total_len = min(len(prompt_tokens) + max_gen_len, params.max_seq_len)   

    # this tokens matrix is to store the input prompts and all the output that is generated by model.
    # later we'll use the tokenizers decode function to decode this token to view results in text format
    tokens = torch.full((batch_size, total_len), fill_value=token_end, dtype=torch.long, device=params.device)

    # fill in the prompt tokens into the token matrix
    tokens[:, :len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=params.device)

    print(prompt)
    # Now we can start inferencing using one token at a time from the prompt_tokens list starting with the first position.
    prev_pos = 0
    for cur_pos in range(1, total_len):
        with torch.no_grad():
            logits, _ = model(x=tokens[:, prev_pos:cur_pos], inference_pos=cur_pos-1)
        if temperature > 0:      
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)        
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)        

        next_token = next_token.reshape(-1)
        if next_token.item() == token_end:
            break

        tokens[:, cur_pos] = next_token

        #print(itos[next_token.item()], end='')
        t = next_token.item()
        print(f"Token {t} '{itos[t]}'")

    #print()

# Perform top-p (nucleus) sampling on a probability distribution.
# probs (torch.Tensor): Probability distribution tensor derived from the logits.
# p: Probability threshold for top-p sampling.
# According to the paper, Top-p sampling selects the smallest set of tokens whose cumulative probability mass exceeds the threshold p. 
# The distribution is renormalized based on the selected tokens.
def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, prob_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(prob_idx, -1, next_token)

device = torch.device('cuda')
model = Transformer(ModelArgs).to(ModelArgs.device)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer, ModelArgs)
