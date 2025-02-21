import torch
import math
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelArgs:
    dim: int = 512              # embedding dimension
    n_layers: int = 8           # number of model decoder blocks
    n_heads: int = 8            # number of heads for queries embedding
    n_kv_heads: int = 4         # number of heads for keys and values embedding
    vocab_size: int = 96        # Length of vocabulary
    multiple_of: int = 256        # Require to calculate dim of feedfoward network
    ffn_dim_multiplier: Optional[float] = None  # Require to calculate dim of feedfoward network
    norm_eps: float = 1e-5                       # Default Epsilon value set for the RMSNorm calculation
    rope_theta: float = 10000.0   # Default theta value for the RePE calculation

    max_batch_size: int = 10     # Max batch size
    max_seq_len: int = 256         # Max sequence length

    epochs: int = 2500             # Total number of training iteration
    log_interval: int = 10        # Number of interval to print the logs and loss values   
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'   # Assign device to cuda or cpu based on availability 


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim).to(device))

    # Shape [batch_size, seq, dim] -> [batch_size, seq, dim]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight

# RoPE
# TODO should default value for theta be vocab_size?
def precompute_freqs_cis(dim: int, seq_len: int, theta: float=10000.0) -> torch.Tensor:
    # Computing Theta value for each dim pair which is dim/2
    device = ModelArgs.device
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2,device=device)[:(dim//2)].float()/dim))

    # Computing range of positions(m) in the sequence
    t = torch.arange(seq_len, dtype=torch.float32, device=device)

    # freqs gives all the Theta value range for all the position of tokens in the sequence
    freqs = torch.outer(t, freqs).to(device)

    # This is the rotation matrix which needs to be converted to Polar form in order to perform rotation to the embedding
    return torch.polar(torch.ones_like(freqs).to(device), freqs).to(device)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), "the last two dimension of freqs_cis, x must match"
    shape = [d if i == 1 or i == ndim-1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = ModelArgs.device
    # Applying rotary positional encoding to both query and key embedding together
    # First: The last dimension of xq and xk embedding needs to be reshaped to make it a pair. As rotation matrix is applied to each pair of dim.
    # Next: convert both xq and xk to complex number as the rotation matrix is only applicable to complex number
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(device)    #xq_:[bsz, seq_len, n_heads, head_dim/2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(device)    #xk_:[bsz, seq_len, n_heads, head_dim/2]

    # The rotation matrix(freqs_cis) dimensions across seq_len(dim=1) and head_dim(dim=3) should match with the embedding
    # Also, the shape freqs_cis should be the same with xq and xk, hence change the shape of freqs_cis:[seq_len,head_dim] -> freqs_cis:[1,seq_len,1,head_dim]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # Finally, perform rotation operation by multiplying with freqs_cis.
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).to(device).type_as(xq) #xq_out:[bsz, seq_len, n_heads, head_dim]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).to(device).type_as(xk) #xk_out:[bsz, seq_len, n_heads, head_dim]
    return xq_out, xk_out


# Group Query Attention with KV Cache
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim  # Embedding dimension
        self.n_heads = args.n_heads  # Number of heads assigned to Query

        # Number of heads assigned to Key and values. If "None", the number will be same as Query.
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Dimension of each head relative to model dimension
        self.head_dim = args.dim // args.n_heads
        # Number of repetition in order to make time Key, Value heads to match Query heads number
        self.n_rep = args.n_heads // args.n_kv_heads

        # Weight initialize for Keys, Querys, Values and Oupt. Notice that the out_feature value of weight for q and kv are based on it's heads
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, device=args.device)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=args.device)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=args.device)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False, device=args.device)

        # Initialize caches to store Key, Values at start. (KV Cache Implementation)
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
            
        self.freqs_cis_inference = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len * 2) # TODO Why x2 for inference?
        self.freqs_cis_training = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len)

    def forward(self, x: torch.Tensor, inference_pos: int | None = None) -> torch.Tensor:
        # Shape of the input embedding: [bsz,seq_len,dim]
        bsz, seq_len, _ = x.shape
        start_pos = inference_pos if inteference_pos is not None else 0
        end_pos = start_pos + seq_len

        xq = self.wq(x)  #x[bsz,seq_len,dim]*wq[dim,n_heads * head_dim] -> q[bsz,seq_len,n_heads * head_dim]
        xk = self.wk(x)  #x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> k[bsz,seq_len,n_kv_heads * head_dim]
        xv = self.wv(x)  #x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> v[bsz,seq_len,n_kv_heads * head_dim]

        # Reshaping Querys, Keys and Values by their number of heads. (Group Query Attention Implementation)
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)      #xq[bsz,seq_len,n_heads, head_dim]
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   #xk[bsz,seq_len,n_kv_heads, head_dim]
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   #xv[bsz,seq_len,n_kv_heads, head_dim]
        
        freqs_cis = self.freqs_cis_inference[start_pos : start_pos + seq_len] if inference else self.freqs_cis_training
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if inference_pos is not None:
            # Store Keys and Values token embedding into their respective cache
            self.cache_k[:bsz, start_pos:end_pos] = xk
            self.cache_v[:bsz, start_pos:end_pos] = xv
            # Assign all the previous tokens embeddings upto current tokens position to Keys and Values variable for Attention Calculation
            xk = self.cache_k[:bsz, :end_pos]
            xv = self.cache_v[:bsz, :end_pos]

        # Use repeat_kv function to make Keys,Values shape same as the queries shape
        xk = repeat_kv(xk, self.n_rep)  #xk[bsz,seq_len,n_heads,head_dim]
        xv = repeat_kv(xv, self.n_rep)  #xv[bsz,seq_len,n_heads,head_dim]

        # To compute attention, we'll need to perform a transpose operation to reshape all queries, keys and values bring heads at dim 1 and seq at dim 2
        xq = xq.transpose(1, 2)  #xq[bsz,n_heads,seq_len,head_dim]
        xk = xk.transpose(1, 2)  #xk[bsz,n_heads,seq_len,head_dim]
        xv = xv.transpose(1, 2)  #xv[bsz,n_heads,seq_len,head_dim]

        # Computing attention score
        scores = torch.matmul(xq, xk.transpose(2, 3)).to(self.args.device) / math.sqrt(self.head_dim)

        # Apply mask during training
        if not inference:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=self.args.device)
            mask = torch.triu(mask, diagonal=1).to(self.args.device)
            scores = scores + mask

        # Apply softmax to the attention score
        # TODO substract max value first, for numerical stability - Does F.softmax() already do it?
        output = F.softmax(scores.float(), dim=-1).type_as(xq) @ xv

        # We get the contextual embedding for each head
        # All heads need to be reshaped back and combined to give a single single contextual attention output
        # Shape change: output[bsz,n_heads,seq_len,head_dim] -> output[bsz,seq_len, n_heads,head_dim] -> output[bsz,seq_len, n_heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        # shape: output [bsz,seq_len,dim]
        return self.wo(output)


# If the number of keys/values heads is less than query heads, this function expands the key/values embeddings with the required number of repetition
def repeat_kv(x:torch.Tensor, n_rep: int)-> torch.Tensor:
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bsz, seq_len, n_kv_heads, n_rep, head_dim).reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: float | None, device: str):
        super().__init__()
        # Models embedding dimension
        self.dim = dim

        # We must use the hidden dimensions calculation shared by Meta which is the ideal one for this model
        # Hidden dimension are calculated such that it is a multiple of 256.
        hidden_dim = int(2 * hidden_dim/3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = ((hidden_dim + multiple_of - 1) // multiple_of) * multiple_of

        # define hidden layer weights
        self.w1 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)
        self.w2 = nn.Linear(hidden_dim, self.dim, bias=False, device=device)
        self.w3 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)

    def forward(self, x):
        # Shape: [batch_size, seq_len, dim]
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps, device=args.device)
        self.attention = Attention(args)
        self.ff_norm = RMSNorm(dim=args.dim, eps=args.norm_eps, device=args.device)
        self.feedforward = FeedForward(dim=args.dim, hidden_dim=4*args.dim, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier, device=args.device)

    def forward(self, x: torch.Tensor, inference_pos: int | None = None):
        # i) pass input embedding to attention_norm and then pass to attention block.
        # ii) the output of attention is then added to embedding(before norm)
        x = x + self.attention(self.attention_norm(x), inference_pos)

        # i) pass attention output to ff_norm and then pass to the feedforward network.
        # ii) the output of feedforward network is then added to the attention output (before ff_norm)
        return x + self.feedforward(self.ff_norm(x))  # Shape: [batch_size, seq_len, dim]


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(args=params))

        self.norm = RMSNorm(params.dim, eps = params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, inference_pos: int | None = None, targets: torch.Tensor | None = None):
        assert (inference_pos is None and targets is not None) or (inference_pos is not None and targets is None)
        # x is the batch of token_ids generated from the texts or prompts using tokenizers.
        # x[bsz, seq_len] -> h[bsz, seq_len, dim]

        x = self.tok_embeddings(x)
        for layer in self.layers:
            x = layer(x, inference_pos)
        x = self.norm(x)

        # h[bsz, seq_len, dim] -> logits[bsz, seq_len, vocab_size]
        logits = self.output(x).float()

        if inference_pos is not None:
            return logits, None

        return logits, F.cross_entropy(logits.view(-1, self.params.vocab_size), targets.view(-1))


if __name__ == "__main__":
    device = 'cuda'
    print("test rms_norm")
    x = torch.randn((ModelArgs.max_batch_size, ModelArgs.max_seq_len, ModelArgs.dim), device=device)
    rms_norm = RMSNorm(dim=ModelArgs.dim)
    x_norm = rms_norm(x)
    print(f"Shape of x: {x.shape}")
    print(f"Shape of x_norm: {x_norm.shape}")
    assert x.shape == torch.Size([10, 256, 512])
    assert x_norm.shape == torch.Size([10, 256, 512])
    
    print("test rope")
    head_dim = ModelArgs.dim//ModelArgs.n_heads
    wq = nn.Linear(ModelArgs.dim, ModelArgs.n_heads * head_dim, bias=False, device=device)
    wk = nn.Linear(ModelArgs.dim, ModelArgs.n_kv_heads * head_dim, bias=False, device=device)
    xq = wq(x_norm)
    xk = wk(x_norm)
    print(f"xq.shape: {xq.shape}")
    print(f"xk.shape: {xk.shape}")
    assert xq.shape == torch.Size([10, 256, 512])
    assert xk.shape == torch.Size([10, 256, 256])

    xq = xq.view(xq.shape[0],xq.shape[1],ModelArgs.n_heads, head_dim)
    xk = xk.view(xk.shape[0],xk.shape[1],ModelArgs.n_kv_heads, head_dim)
    print(f"xq.re-shape: {xq.shape}")
    print(f"xk.re-shape: {xk.shape}")
    assert xq.shape == torch.Size([10, 256, 8, 64])
    assert xk.shape == torch.Size([10, 256, 4, 64])

    freqs_cis = precompute_freqs_cis(dim=head_dim, seq_len=ModelArgs.max_seq_len)
    print(f"freqs_cis.shape: {freqs_cis.shape}")
    assert freqs_cis.shape == torch.Size([256, 32])

    xq_rotate, xk_rotate = apply_rotary_emb(xq, xk, freqs_cis)
    print(f"xq_rotate.shape: {xq_rotate.shape}")
    print(f"xk_rotate.shape: {xk_rotate.shape}")
    assert xq_rotate.shape == torch.Size([10, 256, 8, 64])
    assert xk_rotate.shape == torch.Size([10, 256, 4, 64])

    print("test repeat_kv")
    n_rep = ModelArgs.n_heads // ModelArgs.n_kv_heads
    keys = repeat_kv(xk, n_rep)
    print(f"xk.shape: {xk.shape}")
    print(f"keys.shape: {keys.shape}")
    assert xk.shape == torch.Size([10, 256, 4, 64])
    assert keys.shape == torch.Size([10, 256, 8, 64])

    print("test attention")
    attention = Attention(ModelArgs)
    x_out = attention(x_norm,start_pos=0, inference=False)
    print(f"x_out.shape: {x_out.shape}")
    assert x_out.shape == torch.Size([10, 256, 512])

    print("test feed_forward")
    feed_forward = FeedForward(ModelArgs.dim, 4 * ModelArgs.dim, ModelArgs.multiple_of, ModelArgs.ffn_dim_multiplier, device=device)
    x_out = rms_norm(x_out)
    x_out = feed_forward(x_out)
    print(f"feed forward output: x_out.shape: {x_out.shape}")
    assert x_out.shape == torch.Size([10, 256, 512])

    print("test transformer_block")
    xx = torch.randn((ModelArgs.max_batch_size, ModelArgs.max_seq_len, ModelArgs.dim), device=device)
    transformer_block = TransformerBlock(ModelArgs)
    transformer_block_out = transformer_block(xx, start_pos=0, inference=False)
    print(f"transformer_block_out.shape: {transformer_block_out.shape}")
    assert transformer_block_out.shape == torch.Size([10, 256, 512])

    print("test model")
    model = Transformer(ModelArgs).to(ModelArgs.device)
    print(model)
