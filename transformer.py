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

from model import ModelArgs, Transformer

device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
print("Loading dataset")
data = []

with py7zr.SevenZipFile('validation.csv.7z', mode='r') as archive:
    csv_bytes_io = archive.read()['validation.csv']
    with io.TextIOWrapper(csv_bytes_io, encoding='utf-8', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row['text'])

print("Building vocabulary")
vocab = set()
for line in data:
    vocab |= set(line)
vocab = sorted(list(vocab))
vocab.extend(['<|begin|>', '<|end|>', '<|pad|>'])

# Create a mapping between characters with corresponding integer indexes in vocabulary.
itos = {i:ch for i, ch in enumerate(vocab)}
stoi = {ch:i for i, ch in enumerate(vocab)}
# Tokenizer's encode function: take a string, output a list of integers
def encode_tokens(s: str) -> List[int]:
    return [stoi[ch] for ch in s]
# Tokenizer's decode function: take a list of integers, output a string
def decode_tokens(l: List[int]) -> str:
    return ''.join(itos[i] for i in l)

# Define tensor token variable to be used later during model training
token_begin = torch.tensor([stoi['<|begin|>']], dtype=torch.int, device=device)
token_end = torch.tensor([stoi['<|end|>']], dtype=torch.int, device=device)
token_pad = torch.tensor([stoi['<|pad|>']], dtype=torch.int, device=device)

prompt = "Hello World"
assert decode_tokens(encode_tokens(prompt)) == prompt

print(f"Lenth of shakespeare in character: {len(data)}")
print(f"The vocabulary looks like this: {''.join(vocab)}\n")
print(f"Vocab size: {len(vocab)}")
print(f"encoded_tokens: {encode_tokens(prompt)}")

dataset = torch.tensor(encode_tokens(data[0]), dtype=torch.int).to(ModelArgs.device)
print(f"dataset-shape: {dataset.shape}")

# Define function to generate batches from the given dataset
def get_dataset_batch(data, split, args:ModelArgs):
    seq_len = args.max_seq_len
    batch_size = args.max_batch_size
    device = args.device

    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)): int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]

    batch_data = train
    if split == "val":
        batch_data = val

    if split == "test":
        batch_data = test
  
    # Picking random starting points from the dataset to give random samples for training, validation and testing.
    ix = torch.randint(0, len(batch_data) - seq_len - 3, (batch_size,)).to(device)
    x = torch.stack([torch.cat([token_begin, batch_data[i:i+seq_len-1]]) for i in ix]).long().to(device)
    y = torch.stack([torch.cat([batch_data[i+1:i+seq_len], token_end]) for i in ix]).long().to(device)
    return x, y

### Test: get_dataset function ###
xs, ys = get_dataset_batch(dataset, split="train", args=ModelArgs)
print([(decode_tokens(xs[i].tolist()), decode_tokens(ys[i].tolist())) for i in range(len(xs))])

# Define a evaluate loss function to calculate and store training and validation loss for logging and plotting
@torch.no_grad()
def evaluate_loss(model, args:ModelArgs):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = []
        for _ in range(10):      
            xb, yb = get_dataset_batch(dataset, split, args)
            _, loss = model(x=xb, targets=yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)

    model.train()
    return out

# Define a training function to perform model training
def train(model: Transformer, optimizer, args: ModelArgs) -> None:
    epochs = args.epochs
    log_interval = args.log_interval
    device = args.device
    losses = []   
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        xs, ys = get_dataset_batch(dataset, 'train', args)
        logits, loss = model(x=xs.to(device), targets=ys.to(device))
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model, args)
            losses += [x]            
            print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f}")
            start_time = time.time()
    
    # Print the final validation loss
    print("validation loss: ", losses[-1]['val'])
    torch.save({'config': config.__dict__, 'state_dict': model.state_dict()}, "transformer_lm.pth")


device = torch.device('cuda')
model = Transformer(ModelArgs).to(ModelArgs.device)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer, ModelArgs)

def generate(model: Transformer, prompt: str, params: ModelArgs, max_gen_len: int=500, temperature: float = 0.6, top_p: float = 0.9) -> str:
    # prompt_tokens: List of user input texts or prompts
    # max_gen_len: Maximum length of the generated text sequence.
    # temperature: Temperature value for controlling randomness in sampling. Defaults to 0.6.
    # top_p: Top-p probability threshold for sampling prob output from the logits. Defaults to 0.9.
    # prompt_tokens = [0]
    bsz = 1  #For inferencing, in general user just input one prompt which we'll take it as 1-batch
    prompt_tokens = token_begin.tolist() + encode_tokens(prompt)
    assert len(prompt_tokens) <= params.max_seq_len
    total_len = min(len(prompt_tokens) + max_gen_len, params.max_seq_len)   

    # this tokens matrix is to store the input prompts and all the output that is generated by model.
    # later we'll use the tokenizers decode function to decode this token to view results in text format
    tokens = torch.full((bsz, total_len), fill_value=token_pad.item(), dtype=torch.long, device=params.device) # TODO torch.long is wastefull here! can we use uint16?

    # fill in the prompt tokens into the token matrix
    tokens[:, :len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=params.device)

    #create a prompt_mask_token for later use to identify if the token is a prompt token or a padding token
    # True if it is a prompt token, False if it is a padding token
    input_text_mask = tokens != token_pad.item()

    #now we can start inferencing using one token at a time from the prompt_tokens list starting with the first position.
    prev_pos = 0
    for cur_pos in range(1, total_len):
        with torch.no_grad():
            logits, _ = model(x=tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
        if temperature > 0:      
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)        
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)        

        next_token = next_token.reshape(-1)

        # only replace the token if it's a padding token
        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token

        prev_pos = cur_pos
        if tokens[:,cur_pos] == token_pad.item() and next_token == token_end.item():
            break

    output_tokens, output_texts = [], []    

    for i, toks in enumerate(tokens.tolist()):
        # eos_idx = toks.index(token_end.item())
        if token_end.item() in toks:
            eos_idx = toks.index(token_end.item())
            toks = toks[:eos_idx]

        output_tokens.append(toks)
        output_texts.append(decode(toks))
    return output_tokens, output_texts

# Perform top-p (nucleus) sampling on a probability distribution.
# probs (torch.Tensor): Probability distribution tensor derived from the logits.
# p: Probability threshold for top-p sampling.
# According to the paper, Top-p sampling selects the smallest set of tokens whose cumulative probability mass exceeds the threshold p. 
# The distribution is renormalized based on the selected tokens.
def sample_top_p(probs: torch.Tensor, p: float) -> int:
    probs_sort, prob_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(prob_idx, -1, next_token)    

sys.exit(0)



def do_train():
    print("Loading dataset")
    data = []
    with open('validation.csv', mode='r', encoding='utf-8', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row['text'])
    
    print("Tokenizing")
    vocab = {}
    for line in data:
        for c in line:
            if c not in vocab:
                vocab[c] = True
    print(f"Vocab size is {len(vocab)}")
    config = Config()
    assert config.vocab_size == len(vocab)

    # Enable GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    train(data)


def do_test():
    print("Init CUDA")
    config = Config()
    device = torch.device('cuda')
    print("Load weights from disk")
    checkpoint = torch.load("transformer_lm.pth", map_location=device)
    model = TransformerLM(Config(**checkpoint['config'])).to(device)
    print("Compile model")
    model = torch.compile(model)
    print("Load weights into model")
    model.load_state_dict(checkpoint['state_dict'])
    
    print("Generate")
    prompt = torch.randint(0, config.vocab_size, (1, 1), device=device)
    output = generate(model, config, prompt, max_len=100)
    print("Generated sequence:", output.cpu().numpy())


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        do_train()
    elif len(sys.argv) == 2 and sys.argv[1] == 'test':
        do_test()
    else:
        print("Expected args: train | test")
