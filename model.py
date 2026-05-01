import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bias: bool = True
    # Use Multi-Head Latent Attention instead of standard attention
    use_mla: bool = False
    # Latent dimension for MLA (if None, defaults to head_dim // 2)
    mla_latent_dim: int = None


class CausalSelfAttention(nn.Module):
    """Multiple attention heads in parallel."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        head_dim = config.n_embd // config.n_head
        assert head_dim % 2 == 0, "head dimension must be even for RoPE"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("tril", torch.tril(
            torch.ones((config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = head_dim

        # Precompute RoPE cache up to maximum sequence length.
        inv_freq = 1.0 / \
            (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(config.block_size, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        cos = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1)
        sin = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _rotate_half(self, x):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1)
        return x_rot.flatten(-2)

    def _apply_rope(self, q, k, T):
        cos = self.rope_cos[:T, :].to(
            device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[:T, :].to(
            device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # (B, T, C)
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        q, k = self._apply_rope(q, k, T)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * k.shape[-1] ** -0.5
        att = att.masked_fill_(self.tril[:, :, :T, :T] == 0, -float('inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention - compresses attention to latent space."""

    def __init__(self, config: GPTConfig, latent_dim: int = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Use 1/2 of head_dim as latent dimension if not specified
        if latent_dim is None:
            latent_dim = (config.n_embd // config.n_head) // 2

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.latent_dim = latent_dim

        # Project input to query, key, value in full dimension
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Project Q to latent space
        self.q_to_latent = nn.Linear(self.head_dim, latent_dim)

        # Project K and V to shared latent space
        self.kv_to_latent = nn.Linear(self.head_dim, latent_dim)

        # Project latent output back to full dimension
        self.latent_to_out = nn.Linear(latent_dim, self.head_dim)

        # Pre-computed combined matrix for Q-K interaction (latent_to_out @ latent_to_out.T effect)
        self.W_qk = nn.Linear(latent_dim, latent_dim)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer("tril", torch.tril(
            torch.ones((config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)))

        # Precompute RoPE cache for latent dimension
        inv_freq = 1.0 / \
            (10000 ** (torch.arange(0, latent_dim, 2).float() / latent_dim))
        t = torch.arange(config.block_size, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        cos = torch.repeat_interleave(
            freqs.cos(), repeats=2, dim=-1)[:, :latent_dim]
        sin = torch.repeat_interleave(
            freqs.sin(), repeats=2, dim=-1)[:, :latent_dim]
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _rotate_half(self, x):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1)
        return x_rot.flatten(-2)

    def _apply_rope(self, q, k, T):
        cos = self.rope_cos[:T, :].to(
            device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[:T, :].to(
            device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def forward(self, x):
        B, T, C = x.size()

        # Standard Q, K, V projections
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # (B, T, C)

        # Reshape to multi-head: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Project to latent space: (B, n_head, T, head_dim) -> (B, n_head, T, latent_dim)
        q_latent = self.q_to_latent(q)
        k_latent = self.kv_to_latent(k)
        v_latent = self.kv_to_latent(v)

        # Apply RoPE to latent queries and keys
        q_latent, k_latent = self._apply_rope(q_latent, k_latent, T)

        # Pre-compute query transformation using combined matrix: (B, n_head, T, latent_dim) -> (B, n_head, T, latent_dim)
        q_for_k = self.W_qk(q_latent)

        # Attention in latent space: (B, n_head, T, latent_dim) x (B, n_head, latent_dim, T) -> (B, n_head, T, T)
        att = (q_for_k @ k_latent.transpose(-2, -1)) * self.latent_dim ** -0.5
        att = att.masked_fill_(self.tril[:, :, :T, :T] == 0, -float('inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to latent values: (B, n_head, T, T) x (B, n_head, T, latent_dim) -> (B, n_head, T, latent_dim)
        y_latent = att @ v_latent

        # Project back to full dimension: (B, n_head, T, latent_dim) -> (B, n_head, T, head_dim)
        y = self.latent_to_out(y_latent)

        # Reshape back: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.nn(x)


class Block(nn.Module):
    """Transformer block: attention + feed-forward with residuals."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Choose attention mechanism based on config
        if config.use_mla:
            self.sa = MultiHeadLatentAttention(
                config, latent_dim=config.mla_latent_dim)
        else:
            self.sa = CausalSelfAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        pe = torch.zeros((config.block_size, config.n_embd))
        pos = torch.arange(0, config.block_size).reshape(
            config.block_size, 1)  # (block_size, 1)
        den = torch.exp(torch.arange(0, config.n_embd, step=2)
                        * -math.log(10000) / config.n_embd)  # (n_embd // 2,)
        pe[:, 0::2] = torch.sin(pos * den)
        pe[:, 1::2] = torch.cos(pos * den)
        self.register_buffer("pe", pe)

    def forward(self, x):
        B, T, C = x.shape
        x = x + self.pe[:T, :]
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(
            config.vocab_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, mask=None):
        tok_emb = self.token_embedding_table(idx)
        x = tok_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)

            if mask is not None:
                # Only score assistant tokens; mask shape: (B, T) with values 0/1
                loss = F.cross_entropy(
                    logits_flat, targets_flat, reduction='none')
                mask_flat = mask.view(B*T)
                loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]  # idx (B, T)
            logits, loss = self(idx_cond)  # logits (B, T, C)
            logits = logits[:, -1, :] / temperature  # (B, C)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx
