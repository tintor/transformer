import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import List, Dict, Tuple, Set
import random
import time
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# local packages
from rope import apply_rotary_emb, precompute_freqs_cis


train_sentences = 1000000
num_epochs = 10
embedding_dim = 16
num_warmup_steps = 1000
num_layers = 4
num_heads = 4
batch_size = 1024
compile_mode = "reduce-overhead" # otherwise "default"

peak_learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
flash_attention = True

wandb_log = True
wandb_project = 'transformer_simple'
wandb_run_name = 'run' + str(time.time())

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

def build_vocabulary(data: List[str]) -> Tuple[List[str], Dict[str, int]]:
    vocab_set: Set[str] = set()
    for line in data:
        vocab_set |= set(line)

    vocab: List[str] = sorted(list(vocab_set))
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
    def __init__(self, embedding_dim: int, num_heads: int):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, E = x.size()
        # Project and reshape for multi-head attention
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if flash_attention:
            attn_out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v)
        # Reshape back and apply final projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, E)
        return self.out(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim, num_heads)
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
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, num_heads: int):
        super(LLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embedding_dim, hidden_dim, num_heads) for _ in range(num_layers)])
        self.output = nn.Linear(embedding_dim, vocab_size)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_blocks(x)
        x = self.output(x)
        return x

    def _generate_token(self, tokens: List[int]) -> int:
        # make tokens size a multiple of 16, to avoid compiling for too many different sizes
        e = 16 - (len(tokens) % 16)
        x = torch.tensor(tokens + [token_pad] * e).unsqueeze(0).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            output = self(x)
            # Use the logit at the last non-pad token position
            return torch.argmax(output[:, len(tokens)-1, :]).item()
    

    @torch.inference_mode()
    def generate(self, text: str, ivocab: Dict[str, int], max_new_tokens: int = 10) -> str:
        self.eval()
        tokens = [ivocab[ch] for ch in text]

        for i in range(max_new_tokens):
            new_token = self._generate_token(tokens)
            if new_token == token_end:
                break
            if new_token == token_pad:
                text += '<|pad|>'
                break
            text += vocab[new_token]
            tokens.append(new_token)
    
        return text

    @torch.inference_mode()
    def compute_accuracy(self, loader: DataLoader) -> float:
        model.eval()
        correct = 0
        count = 0

        with torch.autocast("cuda", dtype=torch.float16):
            for x, y in loader:
                out = self(x)
                # Determine lengths for each sequence (ignoring pad tokens)
                lengths = (x != token_pad).sum(dim=1)
                # Get logits of the last non-pad token for each sample
                logits = out[torch.arange(out.size(0)), lengths - 1]
                # Predicted token using argmax
                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                count += x.size(0)

        return correct / count * 100

    @torch.inference_mode()
    def compute_loss_and_accuracy(self, loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        loss_sum = 0.0
        loss_count = 0
        accuracy_sum = 0.0
        accuracy_count = 0

        with torch.autocast("cuda", dtype=torch.float16):
            for x, y in loader:
                out = self(x)
                lengths = (x != token_pad).sum(dim=1)
                logits = out[torch.arange(out.size(0)), lengths - 1]  # shape: (batch, vocab_size)
                loss = self.cross_entropy_loss(logits, y)
                preds = torch.argmax(logits, dim=1)

                loss_sum += loss.item()
                loss_count += 1
                accuracy_sum += (preds == y).sum().item()
                accuracy_count += x.size(0)

        return loss_sum / loss_count, accuracy_sum / accuracy_count * 100

    def train_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16):
            out = self(x)  # x shape: (batch, seq_len, ...)
            lengths = (x != token_pad).sum(dim=1)  # shape: (batch,)
            logits = out[torch.arange(out.size(0)), lengths - 1]  # shape: (batch, vocab_size)
            loss = self.cross_entropy_loss(logits, y)
                
            preds = torch.argmax(logits, dim=1)
            accuracy = torch.sum(preds == y) / x.size(0) * 100

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        return loss.item(), accuracy

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def sample(min_digits: int, max_digits: int) -> int:
    d = random.randint(min_digits, max_digits)
    if d == 0:
        return 0
    return random.randint(10**(d-1), (10**d) - 1)


print("Building dataset")
data = []
for a in range(0, 100):
    for b in range(0, 100):
        sa = str(a)[::-1]
        sb = str(b)[::-1]
        sab = str(a + b)[::-1]
        s = f"{sa}+{sb}={sab}"
        data.append(s)

val_sentences = train_sentences // 9
extra_val_sentences = val_sentences // 9

data_set: Set[str] = set()
while len(data) < train_sentences + val_sentences:
    a = sample(0, 5)
    b = sample(0, 5)
    sa = str(a)[::-1]
    sb = str(b)[::-1]
    sab = str(a + b)[::-1]
    s = f"{sa}+{sb}={sab}"
    # 21 is max length of sentence in extra_val_data
    s = ' ' * random.randint(0, 21 - len(s)) + s
    if s not in data_set:
        data.append(s)
        data_set.union(s)

train_data = data[:train_sentences]
val_data = data[train_sentences:]

extra_val_data: List[str] = []
data_set.clear()
while len(extra_val_data) < extra_val_sentences:
    a = sample(0, 6)
    b = sample(0, 6)
    sa = str(a)[::-1]
    sb = str(b)[::-1]
    if len(sa) <= 4 and len(sb) <= 4:
        continue
    sab = str(a + b)[::-1]
    s = f"{sa}+{sb}={sab}"
    if s not in data_set:
        extra_val_data.append(s)
        data_set.union(s)
del data_set
assert max(len(e) for e in extra_val_data) == max(len(e) for e in data)

def write_data(data: List[str], fname: str) -> None:
    with open(fname, "w") as f:
        for line in data:
            f.write(line + '\n')

write_data(train_data, 'train_data.txt')
write_data(val_data, 'val_data.txt')
write_data(extra_val_data, 'extra_val_data.txt')

vocab, ivocab = build_vocabulary(data)
token_pad = ivocab['<|pad|>']
token_end = ivocab['<|end|>']

def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs).cuda(), torch.tensor(ys, dtype=torch.long, device='cuda')

def tokenize_dataset(data: List[str]) -> List[Tuple]:
    xy = []
    max_size = max(len(sentence) for sentence in data)
    m = np.full((max_size,), token_pad, dtype=np.int64)
    for sentence in tqdm(data):
        eq = sentence.find('=')

        tokenized = np.fromiter((ivocab[ch] for ch in sentence), dtype=np.int64, count=len(sentence))
        for i in range(eq + 1, len(tokenized) + 1):
            m[:i] = tokenized[:i]
            m[i:] = token_pad
            x = torch.tensor(m)  # creates a copy
            y = token_end if i == len(tokenized) else int(tokenized[i])
            xy.append((x, y))
    return xy

def process_sentence(sentence, ivocab, token_pad, token_end, max_size):
    eq = sentence.find('=')
    tokenized = np.array([ivocab[ch] for ch in sentence], dtype=np.int64)
    padded = np.pad(tokenized, (0, max_size - len(tokenized)), constant_values=token_pad)
    results = []
    for i in range(eq+1, len(tokenized)+1):
        arr = padded.copy()
        arr[i:] = token_pad
        y = token_end if i == len(tokenized) else int(tokenized[i])
        results.append((torch.tensor(arr), y))
    return results

def tokenize_dataset_thread(data):
    max_size = max(len(sentence) for sentence in data)
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_sentence, sentence, ivocab, token_pad, token_end, max_size)
                   for sentence in data]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.extend(future.result())
    return results

print("Train tokens")
train_tokens = tokenize_dataset(train_data)
train_loader = DataLoader(train_tokens, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

print("Val tokens")
val_tokens = tokenize_dataset(val_data)
val_loader = DataLoader(val_tokens, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

print("Extra val tokens")
extra_val_tokens = tokenize_dataset(extra_val_data)
extra_val_loader = DataLoader(extra_val_tokens, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

print("Build model")
hidden_dim = embedding_dim * 4
model = LLM(len(vocab), embedding_dim, hidden_dim, num_layers=num_layers, num_heads=num_heads).cuda()
#model = torch.compile(LLM(len(vocab), embedding_dim, hidden_dim, num_layers).cuda(), mode=compile_mode)
scaler = torch.amp.GradScaler()

num_train_tokens = len(train_tokens)
num_training_steps = num_epochs * int(math.ceil(num_train_tokens / batch_size))

def lr_lambda(current_step: int) -> float:
    if current_step < num_warmup_steps:
        # Linear warmup
        return float(current_step) / float(max(1, num_warmup_steps))
    # Cosine decay after warmup
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))

optimizer = optim.AdamW(model.parameters(), lr=peak_learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"Vocab size: {len(vocab)}, Train Sentences: {len(train_data)}, Train Tokens: {len(train_tokens)}, Val Sentences: {len(val_data)}, Val Tokens: {len(val_tokens)}, Model Params: {model.num_params()}")

total_tokens = 0

if wandb_log:
    ts = time.perf_counter()
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    train_loss, train_accuracy = model.compute_loss_and_accuracy(train_loader)
    val_loss, val_accuracy = model.compute_loss_and_accuracy(val_loader)

    wandb.log({
        "tokens": 0,
        "epoch": 0,
        "train/loss": train_loss,
        "train/acc": train_accuracy,
        "val/loss": val_loss,
        "val/acc": val_accuracy,
        "learning_rate": 0,
    })
    extra_val_acc = model.compute_accuracy(extra_val_loader)
    elapsed = time.perf_counter() - ts

    print(f"Initial state", end='')
    print(f", Train Loss {train_loss:.6f}", end='')
    print(f", Train Accuracy {train_accuracy:.2f}%", end='')
    print(f", Elapsed {elapsed:.3f}s", end='')
    print(f", Val Loss {val_loss:.6f}", end='')
    print(f", Val Accuracy {val_accuracy:.2f}%", end='')
    print(f", Extra Val Accuracy {extra_val_acc:.2f}%", end='')
    print()


for epoch in range(1, num_epochs+1):
    ts = time.perf_counter()
    torch_elapsed = 0.0
    loss_sum = 0
    accuracy_sum = 0
    batch_count = 0
    model.train()

    for x, y in train_loader:
        torch_ts = time.perf_counter()
        loss, accuracy = model.train_batch(x, y)
        loss_sum += loss
        accuracy_sum += accuracy
        batch_count += 1
        torch_elapsed += time.perf_counter() - torch_ts
        total_tokens += y.numel()

    train_loss = loss_sum / batch_count
    train_accuracy = accuracy_sum / batch_count
    elapsed = time.perf_counter() - ts

    learning_rate = optimizer.param_groups[0]['lr']
    assert learning_rate > 0
    print(f"Epoch {epoch}, Tokens {total_tokens}", end='')
    print(f" Learning Rate {learning_rate}", end='')
    print(f", Train Loss {train_loss:.6f}", end='')
    print(f", Train Accuracy {train_accuracy:.2f}%", end='')
    print(f", Elapsed {elapsed:.3f}s", end='')
    print(f", Torch Elapsed {torch_elapsed:.3f}s", end='')

    if epoch % 1 == 0:
        ts = time.perf_counter()
        val_loss, val_accuracy = model.compute_loss_and_accuracy(val_loader)
        print(f", Val Loss {val_loss:.6f}", end='')
        print(f", Val Accuracy {val_accuracy:.2f}%", end='')
        extra_val_acc = model.compute_accuracy(extra_val_loader)
        print(f", Extra Val Accuracy {extra_val_acc:.2f}%", end='')
        te = time.perf_counter()
        print(f", Additional Elapsed {te-ts:.3f}s", end='')
        print()

        if wandb_log:
            wandb.log({
                "tokens": total_tokens,
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_accuracy,
                "val/loss": val_loss,
                "val/acc": val_accuracy,
                "extra_val/acc": extra_val_acc,
                "learning_rate": learning_rate,
            })

        for e in '0+5 1+1 5+5 9+9 01+01 99+99 555+555 123+123 0+45678 1+99999'.split(' '):
            print(model.generate(e + '=', ivocab))
        print()
    else:
        print()

# TODO parallelize [loading of next batch] with [training of current batch]
# TODO model checkpointing and resuming
# TODO smaller type than torch.long for tokens
# TODO ROPE
# TODO kv cache
# TODO fine tuning
# TODO RL
