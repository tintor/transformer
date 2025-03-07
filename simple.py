import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import List, Dict, Tuple
import random

def tokenize(text: str, ivocab: Dict[str, int]) -> List[int]:
    return [ivocab[ch] for ch in text]


def build_vocabulary(data: List[str]) -> Tuple[List[str], Dict[str, int]]:
    vocab = set()
    for line in data:
        vocab |= set(line)
    vocab = sorted(list(vocab))
    vocab.extend(['<|end|>'])
    return vocab, {ch:i for i, ch in enumerate(vocab)}


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.bmm(attention_weights, values)
        return attended_values


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(x + attended)
        forwarded = self.feed_forward(x)
        x = self.norm2(x + forwarded)
        return x


class LLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embedding_dim, hidden_dim) for _ in range(num_layers)])
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1) # Transpose for positional encoding
        x = self.positional_encoding(x)
        x = x.transpose(0, 1) # Transpose back
        x = self.transformer_blocks(x)
        x = self.output(x)
        return x


def sample():
    d = random.randint(0, 2)
    if d == 0:
        return 0
    return random.randint(10**(d-1), (10**d) - 1)


data = []
while len(data) < 1000:
    a = sample()
    b = sample()
    sa = str(a)[::-1]
    sb = str(b)[::-1]
    sab = str(a + b)[::-1]
    s = f"{sa}+{sb}={sab}"
    if s not in data:
        data.append(s)

vocab, ivocab = build_vocabulary(data)
token_end = ivocab['<|end|>']

embedding_dim = 16
hidden_dim = embedding_dim * 4
num_layers = 4

model = LLM(len(vocab), embedding_dim, hidden_dim, num_layers).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

tokenized_data = [tokenize(sentence, ivocab) for sentence in data]

dataset = []
for sentence in data:
    eq = sentence.find('=')
    tokenized = tokenize(sentence, ivocab)
    for i in range(eq + 1, len(tokenized) + 1):
        x = torch.tensor(tokenized[:i]).unsqueeze(0).cuda()
        target = token_end if i == len(tokenized) else tokenized[i]
        y = torch.tensor(target).unsqueeze(0).cuda()
        dataset.append((x, y))
        #t = '$' if i == len(tokenized) else sentence[i]
        #print(f"{sentence[:i]} {t}")

def generate(prompt: str) -> str:
    while True:
        prompt_tokens = tokenize(prompt, ivocab)
        input_tensor = torch.tensor(prompt_tokens).unsqueeze(0).cuda()
        output = model(input_tensor)
        out_token = torch.argmax(output[:, -1, :]).item()
        if out_token == token_end:
            break
        prompt += vocab[out_token]
        if len(prompt) > 100:
            prompt += '<|too_long|>'
            break
    return prompt    

for epoch in range(1000):
    random.shuffle(dataset)
    loss_sum = 0
    loss_count = 0
    for x, y in dataset:
        optimizer.zero_grad()
        out = model(x)[:, -1, :]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        loss_count += 1

    print(f"Epoch {epoch+1}, Loss: {loss_sum / loss_count}")
    print(f"{generate('1+1=')}")
    print(f"{generate('5+5=')}")
    print(f"{generate('01+01=')}")
    print(f"{generate('05+05=')}")
    print(f"{generate('55+55=')}")
# TODO validation loss
# TODO mixed precision training
# TODO batched training
