import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import List, Dict, Tuple
import random
import time
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

#torch.backends.cudnn.benchmark = True

num_epochs = 10000
embedding_dim = 16
num_warmup_steps = 1000
num_layers = 2
batch_size = 245 #256
compile_mode = "reduce-overhead" # otherwise "default"

learning_rate = 2e-4
betas = (0.9, 0.95)
weight_decay = 0.1

def tokenize(text: str, ivocab: Dict[str, int]) -> List[int]:
    return [ivocab[ch] for ch in text]


def build_vocabulary(data: List[str]) -> Tuple[List[str], Dict[str, int]]:
    vocab = set()
    for line in data:
        vocab |= set(line)
    vocab = sorted(list(vocab))
    vocab.extend(['<|end|>', '<|pad|>'])
    return vocab, {ch:i for i, ch in enumerate(vocab)}


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_seq_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embedding_dim)
        return x + self.pe[:, :x.size(1), :]


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embedding_dim)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32, device=x.device))
        # Create a causal mask (lower triangular)
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.bmm(attention_weights, values)
        return attended_values


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x)
        x = self.norm1(x + attended)
        forwarded = self.feed_forward(x)
        x = self.norm2(x + forwarded)
        return x


class LLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int):
        super(LLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embedding_dim, hidden_dim) for _ in range(num_layers)])
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch_size, seq_len)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_blocks(x)
        x = self.output(x)
        return x


def sample():
    d = random.randint(0, 3)
    if d == 0:
        return 0
    return random.randint(10**(d-1), (10**d) - 1)


data = []
for a in range(0, 10):
    for b in range(0, 10):
        sa = str(a)[::-1]
        sb = str(b)[::-1]
        sab = str(a + b)[::-1]
        s = f"{sa}+{sb}={sab}"
        data.append(s)

with open("addition.txt", "w") as f:
    for line in data:
        f.write(line + '\n')

#while len(data) < 10000:
#    a = sample()
#    b = sample()
#    sa = str(a)[::-1]
#    sb = str(b)[::-1]
#    sab = str(a + b)[::-1]
#    s = f"{sa}+{sb}={sab}"
#    if s not in data:
#        data.append(s)

vocab, ivocab = build_vocabulary(data)
token_pad = ivocab['<|pad|>']
token_end = ivocab['<|end|>']

hidden_dim = embedding_dim * 4

print("Compiling model")
ts = time.perf_counter()
model = torch.compile(LLM(len(vocab), embedding_dim, hidden_dim, num_layers).cuda(), mode=compile_mode)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler()
if False:
    with torch.autocast('cuda', dtype=torch.float16):
        x = torch.randint(0, len(vocab), (batch_size, 10), dtype=torch.long).cuda()
        y = torch.randint(0, len(vocab), (batch_size,), dtype=torch.long).cuda()
        out = model(x)
        lengths = (x != token_pad).sum(dim=1)  # shape: (batch,)
        out = out[torch.arange(out.size(0)), lengths - 1]  # shape: (batch, vocab_size)
        loss = criterion(out, y)
    scaler.scale(loss).backward()
te = time.perf_counter()
print(f"Compilation took {te-ts:.0f}s")

tokenized_data = [tokenize(sentence, ivocab) for sentence in data]
f = open("xy.txt", "w")
train_tokens = []
for sentence in data:
    eq = sentence.find('=')
    tokenized = tokenize(sentence, ivocab)
    for i in range(eq + 1, len(tokenized) + 1):
        x = torch.tensor(tokenized[:i], dtype=torch.long).cuda()  # shape: (seq_len,)
        target = token_end if i == len(tokenized) else tokenized[i]
        y = torch.tensor(target, dtype=torch.long).cuda()
        train_tokens.append((x, y))

        t = '$' if i == len(tokenized) else sentence[i]
        f.write(f"{sentence[:i]} : {t}\n")
f.close()
num_train_tokens = len(train_tokens)

num_training_steps = num_epochs * int(math.ceil(num_train_tokens / batch_size))

def lr_lambda(current_step: int) -> float:
    if current_step < num_warmup_steps:
        # Linear warmup
        return float(current_step) / float(max(1, num_warmup_steps))
    # Cosine decay after warmup
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

max_context_size = 20

# New collate function to pad sequences and form batches.
def collate_fn(batch):
    xs, ys = zip(*batch)
    # Pad each sequence to exactly max_context_size
    xs = [torch.cat([x, torch.full((max_context_size - len(x),), token_pad).cuda()]) 
          if len(x) < max_context_size else x
          for x in xs]
    return torch.stack(xs).cuda(), torch.stack(ys).cuda()


train_loader = DataLoader(train_tokens, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

@torch.inference_mode()
def generate(prompt: str) -> str:
    ts = time.perf_counter()
    model.eval()
    while True:
        if len(prompt) > max_context_size:
            prompt += '<|too_long|>'
            break

        prompt_tokens = tokenize(prompt, ivocab)
        length = len(prompt_tokens)
        while len(prompt_tokens) < max_context_size:
            prompt_tokens.append(token_pad)

        x = torch.tensor(prompt_tokens).unsqueeze(0).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(x)
            # Use the logit at the last non-pad token position
            out_token = torch.argmax(output[:, length - 1, :]).item()
        if out_token == token_end:
            break
        if out_token == token_pad:
            prompt += '<|pad|>'
            break
        prompt += vocab[out_token]

    te = time.perf_counter()
    return f"{prompt} | {te-ts:.3f}s"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"Vocab size: {len(vocab)}, Train Sentences: {len(data)}, Train Tokens: {len(train_tokens)}, Model Params: {count_parameters(model)}")
#print(model)

@torch.inference_mode()
def compute_accuracy() -> float:
    model.eval()
    correct = 0
    count = 0

    with torch.no_grad():
        for x, y in train_loader:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
                # Determine lengths for each sequence (ignoring pad tokens)
                lengths = (x != token_pad).sum(dim=1)
                # Get logits of the last non-pad token for each sample
                logits = out[torch.arange(out.size(0)), lengths - 1]
                # Predicted token using argmax
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                count += x.size(0)

    return correct / count * 100


for epoch in range(num_epochs):
    ts = time.perf_counter()
    loss_sum = 0
    loss_count = 0
    model.train()

    for x, y in train_loader:
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(x)  # x shape: (batch, seq_len, ...)
            lengths = (x != token_pad).sum(dim=1)  # shape: (batch,)
            out = out[torch.arange(out.size(0)), lengths - 1]  # shape: (batch, vocab_size)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        loss_sum += loss.item()
        loss_count += 1
    te = time.perf_counter()
    elapsed = te - ts
    
    if epoch % 20 == 0:
        acc = compute_accuracy()
        _acc = f"{acc:.2f}%"
    else:
        acc = 0
        _acc = ''

    print(f"Epoch {epoch+1}, Train Loss: {loss_sum / loss_count}, Elapsed: {elapsed:.3f}s, Train Accuracy: {_acc}")
    if epoch % 20 == 0:
        print(f"{generate('0+5=')}")
        print(f"{generate('1+1=')}")
        print(f"{generate('5+5=')}")
        print(f"{generate('9+9=')}")
        print()
        if acc >= 99.99:
            break

# TODO multi head attention (and variants)
# TODO validation loss
# TODO flash attention
# TODO ROPE
# TODO kv cache
# TODO fine tuning
# TODO RL
