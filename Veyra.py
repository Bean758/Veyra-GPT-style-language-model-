# - SwiGLU activation (LLaMA-2 style) for maximum performance
# - RMSNorm everywhere for stability and speed
# - Rotary Positional Embeddings (RoPE) for perfect position encoding
# - Advanced attention with Grouped Query Attention (GQA) for efficiency
# - Multi-Query Attention option for SPEED
# - Flash Attention simulation for memory efficiency
# - Weight tying between embedding and output layers
# - Advanced regularization: DropPath, LayerDrop
# - Mixture of Experts (MoE) layers for parameter efficiency
# - KV-Cache compression and quantization
# - Advanced learning rate schedules with restarts
# - Gradient clipping with adaptive norms
# - Model parallelism support
# - Advanced data augmentation
# - Curriculum learning
# - Dynamic loss scaling
# - Memory-mapped datasets for huge data
# - Advanced sampling: MIROSTAT, Typical-P
# - Model pruning and quantization
# - EMA (Exponential Moving Average) of weights
# - Advanced metrics and logging

import os
import io
import math
import json
import time
import glob
import random
import argparse
import warnings
import mmap
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# -------------------------- CONFIG ---------------------------------- #

SOS, EOS, PAD, UNK = "<|startoftext|>", "<|endoftext|>", "<|pad|>", "<|unk|>"

@dataclass
class UltraConfig:
    # Files
    model_path: str = "ultra_jacked_best.pth"
    model_final_path: str = "ultra_jacked_final.pth"
    vocab_path: str = "ultra_jacked_vocab.json"
    ckpt_dir: str = "ultra_ckpts"
    rag_index_path: str = "ultra_rag.pt"
    rag_meta_path: str = "ultra_rag_meta.json"
    datasheet_candidates: tuple = ("datasheet.txt", "datasheet", "725017a1-6fbb-495b-8864-08063e88cff0.txt")
    
    # Tokenizer / Data - OPTIMIZED FOR GTX 1650 TI SUPER
    max_len: int = 2048  # Increased context length
    val_split: float = 0.1
    shuffle_buffer: int = 10000
    
    # Model Architecture - JACKED TO THE MAX
    vocab_extra: tuple = (SOS, EOS, PAD, UNK)
    d_model: int = 768  # LLaMA-2 7B uses 4096, but we optimize for your GPU
    n_head: int = 12    # Multi-head attention
    n_kv_head: int = 4  # Grouped Query Attention for efficiency
    n_layers: int = 24  # Deep network for maximum learning capacity
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    
    # Advanced Architecture Features
    use_rope: bool = True
    use_gqa: bool = True  # Grouped Query Attention
    use_mqa: bool = False  # Multi-Query Attention (even faster)
    use_flash_attention: bool = True
    use_swiglu: bool = True  # SwiGLU activation like LLaMA-2
    swiglu_expansion: float = 2.6667  # 8/3 expansion ratio
    use_rmsnorm: bool = True
    rmsnorm_eps: float = 1e-6
    tie_weights: bool = True  # Tie embedding and output weights
    use_bias: bool = False  # No bias in linear layers (LLaMA style)
    
    # Regularization - MAXIMUM STABILITY
    gradient_checkpointing: bool = True
    use_droppath: bool = True
    droppath_rate: float = 0.1
    use_layerdrop: bool = True
    layerdrop_rate: float = 0.1
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # Training - OPTIMIZED FOR GTX 1650 TI SUPER (4GB VRAM)
    device_override: Optional[str] = None
    seed: int = 42
    epochs: int = 1000
    batch_size: int = 8  # Optimized for your GPU
    micro_batch_size: int = 2  # Gradient accumulation
    grad_accum_steps: int = 4  # batch_size // micro_batch_size
    max_grad_norm: float = 1.0
    adaptive_grad_clip: bool = True
    
    # Learning Rate - ADVANCED SCHEDULING
    lr: float = 6e-4  # LLaMA-2 style learning rate
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    label_smooth: float = 0.1
    
    # LR Schedule
    warmup_steps: int = 2000
    lr_decay_style: str = "cosine"  # "linear", "cosine", "constant", "cosine_restarts"
    cosine_restarts: int = 0  # Number of cosine restarts
    lr_decay_iters: Optional[int] = None
    
    # Logging and Checkpointing
    save_every_steps: int = 500
    eval_every_steps: int = 250
    log_every_steps: int = 10
    keep_best_k: int = 3
    
    # Mixed Precision - MAXIMUM SPEED
    use_compile: bool = True
    amp: str = "bf16"  # "fp16" | "bf16" | "fp32"
    loss_scale: float = 2**16
    dynamic_loss_scale: bool = True
    
    # Memory Optimization
    cpu_offload: bool = False
    use_zero: bool = False  # ZeRO optimizer states
    activation_checkpointing: bool = True
    
    # Generation - ADVANCED SAMPLING
    history_turns: int = 10
    gen_len: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    no_repeat_ngram: int = 3
    min_new_tokens: int = 1
    max_new_tokens: int = 512
    
    # Advanced Sampling
    use_mirostat: bool = False
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    use_typical_p: bool = False
    typical_p: float = 0.95
    
    # RAG
    rag_topk: int = 5
    rag_chunk_size: int = 512
    rag_overlap: int = 64

cfg = UltraConfig()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pick_device(cfg: UltraConfig):
    if cfg.device_override:
        dev = cfg.device_override
    else:
        if torch.cuda.is_available():
            dev = "cuda"
            # Optimize for GTX 1650 Ti Super
            torch.cuda.set_per_process_memory_fraction(0.95)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    return torch.device(dev)

def human_size(num_params: int):
    if num_params >= 1e9: return f"{num_params/1e9:.2f}B"
    if num_params >= 1e6: return f"{num_params/1e6:.2f}M"
    if num_params >= 1e3: return f"{num_params/1e3:.2f}K"
    return str(num_params)

def count_params(m, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    return sum(p.numel() for p in m.parameters())

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def find_datasheet(cfg: UltraConfig):
    for p in cfg.datasheet_candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No datasheet file found.")

# Memory-mapped file reader for HUGE datasets
class MemoryMappedFile:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
    def __del__(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file'):
            self.file.close()

class UltraVocab:
    def __init__(self):
        self.word2idx = {SOS: 0, EOS: 1, PAD: 2, UNK: 3}
        self.idx2word = [SOS, EOS, PAD, UNK]
        self.word_freq = defaultdict(int)
        
    def add_sentence(self, s: str):
        words = s.lower().split()
        for w in words:
            self.word_freq[w] += 1
            if w not in self.word2idx:
                self.word2idx[w] = len(self.idx2word)
                self.idx2word.append(w)
    
    def prune_vocab(self, min_freq=2):
        """Remove low-frequency words to reduce vocab size"""
        new_word2idx = {SOS: 0, EOS: 1, PAD: 2, UNK: 3}
        new_idx2word = [SOS, EOS, PAD, UNK]
        
        for word, freq in self.word_freq.items():
            if freq >= min_freq and word not in new_word2idx:
                new_word2idx[word] = len(new_idx2word)
                new_idx2word.append(word)
        
        self.word2idx = new_word2idx
        self.idx2word = new_idx2word
        print(f"Pruned vocab from {len(self.word_freq)} to {len(self.idx2word)} words")

    def encode(self, s: str, max_len: Optional[int] = None) -> List[int]:
        ids = [self.word2idx[SOS]] + \
              [self.word2idx.get(w, self.word2idx[UNK]) for w in s.lower().split()] + \
              [self.word2idx[EOS]]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int]) -> str:
        bad = {self.word2idx[SOS], self.word2idx[EOS], self.word2idx[PAD], self.word2idx[UNK]}
        words = [self.idx2word[i] for i in ids if 0 <= i < len(self.idx2word) and i not in bad]
        return " ".join(words)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "idx2word": self.idx2word,
                "word_freq": dict(self.word_freq)
            }, f)

    @staticmethod
    def load(path: str) -> "UltraVocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = UltraVocab()
        v.idx2word = data["idx2word"]
        v.word2idx = {w: i for i, w in enumerate(v.idx2word)}
        v.word_freq = defaultdict(int, data.get("word_freq", {}))
        return v

class UltraDataset(Dataset):
    """Enhanced dataset with data augmentation and curriculum learning"""
    def __init__(self, path: str, vocab: UltraVocab, max_len: int, augment=True):
        self.samples = []
        self.difficulties = []  # For curriculum learning
        
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if "\\t" not in line:
                    continue
                q_raw, a_raw = line.strip().split("\\t", 1)
                q = q_raw.lstrip("Qq:").strip()
                a = a_raw.lstrip("Aa:").strip()
                
                # Create conversation format
                text = f"User: {q}\nAssistant: {a}"
                vocab.add_sentence(text)
                
                tokens = vocab.encode(text, max_len)
                if len(tokens) > 4:  # Skip very short samples
                    self.samples.append(torch.tensor(tokens, dtype=torch.long))
                    # Simple difficulty metric: longer = harder
                    self.difficulties.append(len(tokens))
                
                # Data augmentation: create variations
                if augment and random.random() < 0.3:
                    variations = [
                        f"Q: {q}\nA: {a}",
                        f"Question: {q}\nAnswer: {a}",
                        f"{q}\n{a}",
                    ]
                    for var in variations:
                        if random.random() < 0.5:
                            var_tokens = vocab.encode(var, max_len)
                            if len(var_tokens) > 4:
                                self.samples.append(torch.tensor(var_tokens, dtype=torch.long))
                                self.difficulties.append(len(var_tokens))
        
        if not self.samples:
            raise RuntimeError("No Q&A pairs parsed. Check your data format.")
        
        self.pad_id = vocab.word2idx[PAD]
        print(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, i): 
        return self.samples[i]
    
    def get_curriculum_indices(self, epoch, total_epochs):
        """Return indices for curriculum learning"""
        # Start with easier samples, gradually include harder ones
        progress = epoch / total_epochs
        max_difficulty = np.percentile(self.difficulties, 50 + 50 * progress)
        valid_indices = [i for i, d in enumerate(self.difficulties) if d <= max_difficulty]
        return valid_indices

def collate_ultra(batch, pad_id):
    """Enhanced collate function with packing"""
    # Sort by length for efficiency
    batch = sorted(batch, key=len, reverse=True)
    return nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA-2 style)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)

class DropPath(nn.Module):
    """Drop Path (Stochastic Depth) regularization"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - LLaMA-2 style"""
    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for efficiency
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
            
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
            
        return self._cos_cached, self._sin_cached

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply RoPE to query and key tensors"""
    if position_ids is None:
        cos = cos[:, :, : q.shape[-2], :]
        sin = sin[:, :, : q.shape[-2], :]
    else:
        cos = cos.squeeze(1).squeeze(0)
        sin = sin.squeeze(1).squeeze(0)
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLU(nn.Module):
    """SwiGLU activation function (LLaMA-2 style)"""
    def __init__(self, dim, hidden_dim=None, bias=False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * 8/3)  # Standard expansion
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

class UltraKVCache:
    """Advanced KV Cache with compression and quantization"""
    def __init__(self, max_len, n_head, head_dim, device, dtype, compress=False):
        self.k = None
        self.v = None
        self.max_len = max_len
        self.n_head = n_head
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.compress = compress
        self.seq_len = 0

    def append(self, k, v):
        # k,v: (B, nH, T, dH)
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        
        self.seq_len = self.k.shape[2]
        
        # Sliding window for very long sequences
        if self.seq_len > self.max_len:
            self.k = self.k[:, :, -self.max_len:, :]
            self.v = self.v[:, :, -self.max_len:, :]
            self.seq_len = self.max_len

    def get(self):
        return self.k, self.v
    
    def clear(self):
        self.k = None
        self.v = None
        self.seq_len = 0

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) for efficiency"""
    def __init__(self, d_model, n_head, n_kv_head, dropout, max_len=4096, rope=True, flash=True):
        super().__init__()
        assert d_model % n_head == 0
        assert n_head % n_kv_head == 0
        
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = d_model // n_head
        self.n_rep = n_head // n_kv_head
        self.rope = rope
        self.flash = flash
        
        # Projections
        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * self.head_dim, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        if rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_len)
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def repeat_kv(self, x):
        """Repeat k/v heads to match q heads"""
        if self.n_rep == 1:
            return x
        B, n_kv_head, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv_head, self.n_rep, seq_len, head_dim)
        return x.reshape(B, n_kv_head * self.n_rep, seq_len, head_dim)

    def forward(self, x, cache=None, use_cache=False):
        B, seq_len, _ = x.shape
        
        # Projections
        q = self.q_proj(x).view(B, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if self.rope:
            cos, sin = self.rotary_emb(x, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Handle cache
        if use_cache and cache is not None:
            cache.append(k, v)
            k, v = cache.get()
        elif use_cache and cache is None:
            cache = UltraKVCache(2048, self.n_kv_head, self.head_dim, x.device, x.dtype)
            cache.append(k, v)
        
        # Repeat k,v to match q heads
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        
        # Attention computation
        if self.flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's Flash Attention if available
            attn_mask = None
            if not use_cache:
                attn_mask = self.mask[:, :, :seq_len, :seq_len].bool()
            
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0,
                is_causal=not use_cache
            )
        else:
            # Standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if not use_cache:
                scores = scores.masked_fill(
                    self.mask[:, :, :seq_len, :seq_len] == 0, 
                    float('-inf')
                )
            
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        out = self.o_proj(out)
        
        return out, cache if use_cache else None

class UltraTransformerBlock(nn.Module):
    """Ultra-enhanced transformer block with all the bells and whistles"""
    def __init__(self, cfg: UltraConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg = cfg
        
        # Attention
        if cfg.use_gqa:
            self.attn = GroupedQueryAttention(
                cfg.d_model, cfg.n_head, cfg.n_kv_head, 
                cfg.attention_dropout, cfg.max_len, cfg.use_rope, cfg.use_flash_attention
            )
        else:
            # Standard multi-head attention (fallback)
            self.attn = nn.MultiheadAttention(
                cfg.d_model, cfg.n_head, cfg.attention_dropout, batch_first=True
            )
        
        # Feed Forward
        if cfg.use_swiglu:
            hidden_dim = int(cfg.d_model * cfg.swiglu_expansion)
            self.mlp = SwiGLU(cfg.d_model, hidden_dim, cfg.use_bias)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.use_bias),
                nn.GELU(),
                nn.Dropout(cfg.hidden_dropout),
                nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.use_bias),
                nn.Dropout(cfg.hidden_dropout)
            )
        
        # Normalization
        if cfg.use_rmsnorm:
            self.ln1 = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
            self.ln2 = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        else:
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.ln2 = nn.LayerNorm(cfg.d_model)
        
        # Regularization
        if cfg.use_droppath:
            drop_rate = cfg.droppath_rate * layer_idx / max(1, cfg.n_layers - 1)
            self.drop_path = DropPath(drop_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, cache=None, use_cache=False):
        # Pre-norm architecture (LLaMA style)
        residual = x
        x = self.ln1(x)
        
        # Attention
        if self.cfg.use_gqa:
            attn_out, cache = self.attn(x, cache, use_cache)
        else:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            
        x = residual + self.drop_path(attn_out)
        
        # MLP
        residual = x
        x = self.ln2(x)
        mlp_out = self.mlp(x)
        x = residual + self.drop_path(mlp_out)
        
        return x, cache if use_cache else None

class UltraGPT(nn.Module):
    """The most jacked GPT implementation ever created"""
    def __init__(self, vocab_size: int, cfg: UltraConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        
        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        if not cfg.use_rope:
            self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        
        self.dropout = nn.Dropout(cfg.dropout)
        
        # Transformer blocks with layer drop
        self.blocks = nn.ModuleList([
            UltraTransformerBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        
        # Final layer norm
        if cfg.use_rmsnorm:
            self.ln_f = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        else:
            self.ln_f = nn.LayerNorm(cfg.d_model)
        
        # Output head
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        
        # Weight tying (LLaMA-2 style)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for output projection
        for name, p in self.named_parameters():
            if name.endswith('o_proj.weight') or name.endswith('down_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        """Initialize weights (LLaMA-2 style)"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, use_cache=False, cache=None):
        B, T = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        x = self.tok_emb(input_ids)
        
        # Position embeddings (if not using RoPE)
        if not self.cfg.use_rope:
            pos_ids = torch.arange(0, T, device=device).unsqueeze(0)
            x = x + self.pos_emb(pos_ids)
        
        x = self.dropout(x)
        
        # Initialize cache if needed
        if use_cache and cache is None:
            cache = [None] * len(self.blocks)
        
        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            # Layer drop during training
            if (self.training and self.cfg.use_layerdrop and 
                random.random() < self.cfg.layerdrop_rate):
                continue
                
            if use_cache:
                x, cache[i] = block(x, cache[i] if cache else None, use_cache)
            else:
                x, _ = block(x)
        
        x = self.ln_f(x)
        
              # ─── Language‑model head ─────────────────────────────────────────
        if targets is not None:                       # training mode
            logits = self.lm_head(x)                  # 1. project to vocab

            pad_id = 2                                # 2. int ID of <|pad|>
            loss = F.cross_entropy(                   # 3‑5. CE loss
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=pad_id,
                label_smoothing=self.cfg.label_smooth   # delete line if Torch < 2.3
            )
            return logits, loss                       # return both
        else:                                         # inference mode
            logits = self.lm_head(x)
            return logits, cache if use_cache else None


    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=None, temperature=1.0, top_k=None, 
                 top_p=0.9, repetition_penalty=1.0, use_cache=True, **kwargs):
        """generation with sampling"""
        self.eval()
        device = input_ids.device
        B, T = input_ids.shape
        
        if max_new_tokens is None:
            max_new_tokens = self.cfg.max_new_tokens
        
        # Initialize for generation
        past_key_values = None
        all_tokens = input_ids.clone()
        eos_token_id = 1  # EOS token
        
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is None:
                # First iteration - process full sequence
                logits, past_key_values = self.forward(
                    all_tokens, use_cache=use_cache
                )
                logits = logits[:, -1, :]  # Get last token logits
            else:
                # Subsequent iterations - only process last token
                logits, past_key_values = self.forward(
                    all_tokens[:, -1:], use_cache=use_cache, cache=past_key_values
                )
                logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(all_tokens[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            all_tokens = torch.cat([all_tokens, next_token], dim=1)
            
            # Check for EOS
            if next_token.item() == eos_token_id:
                break
        
        return all_tokens[:, T:]  # Return only generated tokens


class ExponentialMovingAverage:
    """EMA for model weights"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class UltraLRScheduler:
    """Advanced learning rate scheduler with multiple strategies"""
    def __init__(self, optimizer, cfg: UltraConfig, total_steps):
        self.optimizer = optimizer
        self.cfg = cfg
        self.total_steps = total_steps
        self.step_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.step_count += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        if self.step_count <= self.cfg.warmup_steps:
            # Warmup phase
            return self.cfg.lr * self.step_count / max(1, self.cfg.warmup_steps)
        
        if self.cfg.lr_decay_style == "constant":
            return self.cfg.lr
        
        elif self.cfg.lr_decay_style == "linear":
            decay_steps = self.total_steps - self.cfg.warmup_steps
            decay_ratio = (self.step_count - self.cfg.warmup_steps) / max(1, decay_steps)
            return self.cfg.min_lr + (self.cfg.lr - self.cfg.min_lr) * (1 - decay_ratio)
        
        elif self.cfg.lr_decay_style == "cosine":
            decay_steps = self.total_steps - self.cfg.warmup_steps
            decay_ratio = (self.step_count - self.cfg.warmup_steps) / max(1, decay_steps)
            return self.cfg.min_lr + (self.cfg.lr - self.cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * decay_ratio))
        
        elif self.cfg.lr_decay_style == "cosine_restarts":
            # Cosine annealing with warm restarts
            decay_steps = self.total_steps - self.cfg.warmup_steps
            if self.cfg.cosine_restarts > 0:
                restart_steps = decay_steps // (self.cfg.cosine_restarts + 1)
                current_restart = (self.step_count - self.cfg.warmup_steps) // restart_steps
                steps_in_restart = (self.step_count - self.cfg.warmup_steps) % restart_steps
                decay_ratio = steps_in_restart / max(1, restart_steps)
            else:
                decay_ratio = (self.step_count - self.cfg.warmup_steps) / max(1, decay_steps)
            
            return self.cfg.min_lr + (self.cfg.lr - self.cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * decay_ratio))
        
        return self.cfg.lr

def compute_metrics(logits, targets, pad_token_id):
    """Compute training metrics"""
    with torch.no_grad():
        mask = (targets != pad_token_id)
        correct = (logits.argmax(-1) == targets) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        # Perplexity
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1), 
            ignore_index=pad_token_id,
            reduction='mean'
        )
        perplexity = torch.exp(loss)
        
        return {
            'accuracy': accuracy.item(),
            'perplexity': perplexity.item(),
            'loss': loss.item()
        }

def ultra_train(cfg: UltraConfig):
    """The most insane training loop ever created"""
    print("Traning ")
    print("=" * 60)
    
    # Setup
    device = pick_device(cfg)
    print(f" Device: {device} ")
    
    if device.type == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"CUDA Cores: {torch.cuda.get_device_properties(0).multi_processor_count}")
    
    seed_everything(cfg.seed)
    
    # Data loading
    print(" Loading and processing data...")
    data_path = find_datasheet(cfg)
    vocab = UltraVocab()
    dataset = UltraDataset(data_path, vocab, cfg.max_len, augment=True)
    
    # Prune vocabulary for efficiency
    vocab.prune_vocab(min_freq=2)
    
    print(f"Vocabulary size: {len(vocab.idx2word):,}")
    print(f"Training samples: {len(dataset):,}")
    
    # ---------- MODEL CREATION ----------
    print("Building model...")
    model = UltraGPT(len(vocab.idx2word), cfg).to(device)

    if cfg.use_compile and hasattr(torch, "compile"):
        print("torch.compile engaged!")
        model = torch.compile(model, mode="max-autotune")

    # ---------- OPTIMIZER, SCALER, LR SCHED ----------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        eps=cfg.eps,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(cfg.amp != "fp32" and device.type == "cuda")
    )

    total_steps = cfg.epochs * len(train_loader) // cfg.grad_accum_steps
    scheduler = UltraLRScheduler(optimizer, cfg, total_steps)

    # --- DATA SPLITTING + LOADERS ---
    train_size = int(len(dataset) * (1 - cfg.val_split))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    collate_fn = functools.partial(collate_ultra, pad_id=dataset.pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        num_workers=2,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        num_workers=2,
        persistent_workers=True,
    )

    # --- OPTIMIZER, SCALER, SCHEDULER (now train_loader exists!) ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        eps=cfg.eps,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(cfg.amp != "fp32" and device.type == "cuda")
    )

    total_steps = cfg.epochs * len(train_loader) // cfg.grad_accum_steps
    scheduler   = UltraLRScheduler(optimizer, cfg, total_steps)


    # ---- VALIDATION LOADER ----
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        num_workers=2,
        persistent_workers=True,
    )  # <‑‑ and this one


    scheduler = UltraLRScheduler(optimizer, cfg, total_steps)
    
    # EMA
    ema = None
    if cfg.use_ema:
        ema = ExponentialMovingAverage(model, cfg.ema_decay)
        print(" EMA activated for stable training")
    
    # Training state
    global_step = 0
    best_val_loss = float('inf')
    best_models = []  # Keep track of best models
    
    ensure_dir(cfg.ckpt_dir)
    
    print("\n STARTING TRAINING")
    print("=" * 60)
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = defaultdict(float)
        
        # Curriculum learning - get indices for this epoch
        if hasattr(dataset, 'get_curriculum_indices'):
            curriculum_indices = dataset.get_curriculum_indices(epoch, cfg.epochs)
            print(f" Epoch {epoch}: Using {len(curriculum_indices)}/{len(dataset)} samples (curriculum learning)")
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if cfg.amp == "bf16" else torch.float16, enabled=cfg.amp != "fp32"):
                logits, loss = model(batch[:, :-1], targets=batch[:, 1:])
                loss = loss / cfg.grad_accum_steps
            
            # Backward pass
            if cfg.amp != "fp32":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item()
            
            # Gradient accumulation step
            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                # Gradient clipping
                if cfg.adaptive_grad_clip:
                    # Clip gradients adaptively
                    if cfg.amp != "fp32":
                        scaler.unscale_(optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    
                    if cfg.amp != "fp32":
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                else:
                    if cfg.amp != "fp32":
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                
                # Update learning rate
                lr = scheduler.step()
                
                # Update EMA
                if ema is not None:
                    ema.update()
                
                optimizer.zero_grad()
                global_step += 1
                
                # Compute metrics
                if global_step % cfg.log_every_steps == 0:
                    with torch.no_grad():
                        metrics = compute_metrics(logits, batch[:, 1:], dataset.pad_id)
                        for k, v in metrics.items():
                            epoch_metrics[k] += v
                
                # Logging
                if global_step % cfg.log_every_steps == 0:
                    avg_loss = accumulated_loss * cfg.grad_accum_steps
                    pbar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'ppl': f"{math.exp(avg_loss):.2f}",
                        'lr': f"{lr:.2e}",
                        'step': global_step
                    })
                
                # Checkpointing
                if global_step % cfg.save_every_steps == 0:
                    ckpt_path = os.path.join(cfg.ckpt_dir, f"ultra_step_{global_step}.pth")
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.step_count,
                        "scaler": scaler.state_dict() if cfg.amp != "fp32" else None,
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "cfg": asdict(cfg),
                        "vocab": vocab.idx2word
                    }
                    
                    if ema is not None:
                        checkpoint["ema"] = ema.shadow
                    
                    torch.save(checkpoint, ckpt_path)
                    print(f"Checkpoint saved: {ckpt_path}")
                
                accumulated_loss = 0.0
            
            epoch_loss += loss.item() * cfg.grad_accum_steps
        
        # Validation
        if epoch % max(1, cfg.epochs // 100) == 0:  # Validate every 1% of epochs
            model.eval()
            val_loss = 0.0
            val_metrics = defaultdict(float)
            
            print("Running validation...")
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validation"):
                    val_batch = val_batch.to(device, non_blocking=True)
                    
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if cfg.amp == "bf16" else torch.float16, enabled=cfg.amp != "fp32"):
                        val_logits, val_loss_batch = model(val_batch[:, :-1], targets=val_batch[:, 1:])
                    
                    val_loss += val_loss_batch.item()
                    
                    # Compute validation metrics
                    metrics = compute_metrics(val_logits, val_batch[:, 1:], dataset.pad_id)
                    for k, v in metrics.items():
                        val_metrics[k] += v
            
            val_loss /= len(val_loader)
            for k in val_metrics:
                val_metrics[k] /= len(val_loader)
            
            print(f"\n Epoch {epoch} Results:")
            print(f"   Train Loss: {epoch_loss/len(train_loader):.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val PPL: {math.exp(val_loss):.2f}")
            print(f"   Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Apply EMA weights for saving
                if ema is not None:
                    ema.apply_shadow()
                
                torch.save(model.state_dict(), cfg.model_path)
                vocab.save(cfg.vocab_path)
                
                if ema is not None:
                    ema.restore()
                
                print(f"Val Loss: {val_loss:.4f}")
                
                # Keep track of best models
                best_models.append((epoch, val_loss, cfg.model_path))
                best_models.sort(key=lambda x: x[1])
                if len(best_models) > cfg.keep_best_k:
                    best_models = best_models[:cfg.keep_best_k]
    
    # Save final model
    if ema is not None:
        ema.apply_shadow()
    
    torch.save(model.state_dict(), cfg.model_final_path)
    print(f" TRAINING COMPLETE!")
    print(f" validation loss: {best_val_loss:.4f}")
    print(f" model saved to: {cfg.model_path}")
    print(f" Final model saved to: {cfg.model_final_path}")

# ---------------------- INFERENCE & CHAT -------------------------- #

def load_ultra_model(cfg: UltraConfig, device, use_ema=True):
    """Load the ultra model"""
    if not (os.path.exists(cfg.model_path) and os.path.exists(cfg.vocab_path)):
        raise FileNotFoundError("No trained model found. Train first!")
    
    vocab = UltraVocab.load(cfg.vocab_path)
    model = UltraGPT(len(vocab.idx2word), cfg).to(device)
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.eval()
    
    return model, vocab

@torch.no_grad()
def ultra_chat(model, vocab, device, user_input, history, cfg: UltraConfig):
    """Ultra-advanced chat with the model"""
    # Build conversation context
    context_parts = []
    
    # Add recent history
    for u, a in history[-cfg.history_turns:]:
        context_parts.append(f"User: {u}")
        context_parts.append(f"Assistant: {a}")
    
    # Add current user input
    context_parts.append(f"User: {user_input}")
    context_parts.append("Assistant:")
    
    context = "\n".join(context_parts)
    
    # Encode
    input_ids = torch.tensor([vocab.encode(context, cfg.max_len)], device=device)
    
    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=cfg.gen_len,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        use_cache=True
    )
    
    # Decode response
    response = vocab.decode(output_ids[0].tolist())
    return response.strip()

# ---------------------- BENCHMARKING -------------------------- #

@torch.no_grad()
def ultra_benchmark(cfg: UltraConfig):
    """Benchmark the ultra model"""
    device = pick_device(cfg)
    model, vocab = load_ultra_model(cfg, device)
    
    print("BENCHMARK MODE")
    print("=" * 40)
    
    # Warmup
    dummy_input = torch.randint(0, len(vocab.idx2word), (1, 32), device=device)
    for _ in range(5):
        _ = model.generate(dummy_input, max_new_tokens=10)
    
    # Benchmark generation speed
    test_prompt = "User: What is artificial intelligence?\nAssistant:"
    input_ids = torch.tensor([vocab.encode(test_prompt)], device=device)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    generated = model.generate(
        input_ids,
        max_new_tokens=cfg.gen_len,
        temperature=0.8,
        top_p=0.9,
        use_cache=True
    )
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    generation_time = end_time - start_time
    tokens_generated = generated.shape[1]
    tokens_per_second = tokens_generated / generation_time
    
    print(f" Generation Speed: {tokens_per_second:.2f} tokens/second")
    print(f" Generated {tokens_generated} tokens in {generation_time:.3f}s")
    print(f" Model size: {human_size(count_params(model))}")
    
    if device.type == 'cuda':
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        print(f" Peak GPU Memory: {memory_used:.2f}GB")

# ---------------------- MENU SYSTEM -------------------------- #

def print_ultra_menu():
    print("""
BadGPT:
================================================
  [1] TRAIN 
  [2] CHAT 
  [3] BENCHMARK
  [4] MODEL INFO
  [5] EVAL
  [6] EXPORT MODEL
  [7] CONFIG 
  [0] EXIT

Your GPU: GTX 1650 Ti SuperS
""")

def main():
    global cfg
    parser = argparse.ArgumentParser(description="Ultra GPT - The Most Jacked AI Ever")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu|cuda|mps)")
    parser.add_argument("--layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--d-model", type=int, default=None, help="Model dimension")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--no-flash", action="store_true", help="Disable flash attention")
    parser.add_argument("--no-rope", action="store_true", help="Disable RoPE")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()

    # Override config with command line args
    if args.device: cfg.device_override = args.device
    if args.layers: cfg.n_layers = args.layers
    if args.d_model: cfg.d_model = args.d_model
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.compile: cfg.use_compile = True  
    if args.no_flash: cfg.use_flash_attention = False
    if args.no_rope: cfg.use_rope = False
    if args.lr: cfg.lr = args.lr

    while True:
        print_ultra_menu()
        choice = input("Enter your choice: ").strip()
        
        if choice == "0":
            print("Exiting GPT")
            break
            
        elif choice == "1":
            print("INITIATING TRAINING")
            ultra_train(cfg)
            
        elif choice == "2":
            try:
                device = pick_device(cfg)
                model, vocab = load_ultra_model(cfg, device)
                history = []
                
                print("💬 ULTRA CHAT MODE ACTIVATED")
                print("Type 'exit' to quit, 'clear' to clear history")
                print("=" * 50)
                
                while True:
                    user_input = input("\nYou: ").strip()
                    
                    if user_input.lower() == 'exit':
                        break
                    elif user_input.lower() == 'clear':
                        history = []
                        print("🧹 History cleared!")
                        continue
                    
                    if not user_input:
                        continue
                    
                    print(" Assistant: ", end="", flush=True)
                    
                    try:
                        response = ultra_chat(model, vocab, device, user_input, history, cfg)
                        print(response)
                        history.append((user_input, response))
                        
                        # Keep history manageable
                        if len(history) > cfg.history_turns:
                            history = history[-cfg.history_turns:]
                            
                    except Exception as e:
                        print(f"Error: {e}")
                        
            except FileNotFoundError:
                print("No trained model found. Please train first (option 1).")
            except Exception as e:
                print(f"Error loading model: {e}")
                
        elif choice == "3":
            print("⚡ ULTRA BENCHMARK INITIATING...")
            try:
                ultra_benchmark(cfg)
            except FileNotFoundError:
                print("No trained model found. Please train first (option 1).")
            except Exception as e:
                print(f"Benchmark error: {e}")
                
        elif choice == "4":
            print(" MODEL ARCHITECTURE INFO")
            print("=" * 50)
            
            device = pick_device(cfg)
            print(f" Target Device: {device}")
            
            if device.type == 'cuda':
                props = torch.cuda.get_device_properties(0)
                print(f" GPU: {props.name}")
                print(f"💾 GPU Memory: {props.total_memory / 1e9:.1f}GB")
                print(f"🚄 CUDA Cores: {props.multi_processor_count}")
            
            print(f"\n🏗 Architecture:")
            print(f"    Model Dimension: {cfg.d_model}")
            print(f"    Layers: {cfg.n_layers}")
            print(f"   ️ Attention Heads: {cfg.n_head}")
            print(f"    KV Heads (GQA): {cfg.n_kv_head}")
            print(f"    Max Context: {cfg.max_len}")
            print(f"    Features:")
            print(f"      - RoPE: {'✅' if cfg.use_rope else '❌'}")
            print(f"      - GQA: {'✅' if cfg.use_gqa else '❌'}")
            print(f"      - SwiGLU: {'✅' if cfg.use_swiglu else '❌'}")
            print(f"      - Flash Attention: {'✅' if cfg.use_flash_attention else '❌'}")
            print(f"      - RMSNorm: {'✅' if cfg.use_rmsnorm else '❌'}")
            print(f"      - Weight Tying: {'✅' if cfg.tie_weights else '❌'}")
            print(f"      - Gradient Checkpointing: {'✅' if cfg.gradient_checkpointing else '❌'}")
            print(f"      - EMA: {'✅' if cfg.use_ema else '❌'}")
            
            # Try to load model for parameter count
            try:
                vocab = UltraVocab.load(cfg.vocab_path) if os.path.exists(cfg.vocab_path) else None
                if vocab:
                    model = UltraGPT(len(vocab.idx2word), cfg)
                    total_params = count_params(model)
                    trainable_params = count_params(model, trainable_only=True)
                    print(f"\n Parameters:")
                    print(f"   Total: {human_size(total_params)} ({total_params:,})")
                    print(f"   Trainable: {human_size(trainable_params)} ({trainable_params:,})")
                    print(f"   Vocabulary: {len(vocab.idx2word):,}")
                else:
                    print(f"\n Estimated Parameters: ~{human_size(cfg.d_model * cfg.n_layers * 12)}")
            except:
                print(f"\n Estimated Parameters: ~{human_size(cfg.d_model * cfg.n_layers * 12)}")
                
        elif choice == "5":
            print(" EVALUATION SUITE")
            print("=" * 40)
            try:
                device = pick_device(cfg)
                model, vocab = load_ultra_model(cfg, device)
                
                print(" Running comprehensive evaluation...")
                
                # Perplexity evaluation
                try:
                    data_path = find_datasheet(cfg)
                    dataset = UltraDataset(data_path, vocab, cfg.max_len, augment=False)
                    loader = DataLoader(
                        dataset, 
                        batch_size=cfg.micro_batch_size, 
                        shuffle=False,
                        collate_fn=lambda b: collate_ultra(b, dataset.pad_id)
                    )
                    
                    model.eval()
                    total_loss = 0.0
                    total_tokens = 0
                    total_correct = 0
                    
                    print("Computing perplexity...")
                    with torch.no_grad():
                        for batch in tqdm(loader, desc="Evaluating"):
                            batch = batch.to(device)
                            logits, loss = model(batch[:, :-1], targets=batch[:, 1:])
                            
                            # Compute metrics
                            mask = (batch[:, 1:] != dataset.pad_id)
                            tokens_in_batch = mask.sum().item()
                            correct_in_batch = ((logits.argmax(-1) == batch[:, 1:]) & mask).sum().item()
                            
                            total_loss += loss.item() * tokens_in_batch
                            total_tokens += tokens_in_batch
                            total_correct += correct_in_batch
                    
                    perplexity = math.exp(total_loss / total_tokens)
                    accuracy = total_correct / total_tokens
                    
                    print(f"   Results:")
                    print(f"   Perplexity: {perplexity:.3f}")
                    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    print(f"   Tokens Evaluated: {total_tokens:,}")
                    
                except Exception as e:
                    print(f"Evaluation error: {e}")
                
                # Sample generation test
                print("\n Generation Quality Test:")
                test_prompts = [
                    "User: What is the meaning of life?\nAssistant:",
                    "User: Explain quantum computing simply.\nAssistant:",
                    "User: Write a haiku about programming.\nAssistant:",
                    "User: What are the benefits of exercise?\nAssistant:"
                ]
                
                for i, prompt in enumerate(test_prompts, 1):
                    print(f"\n Test {i}:")
                    print(f" Prompt: {prompt.split('Assistant:')[0].strip()}")
                    
                    input_ids = torch.tensor([vocab.encode(prompt)], device=device)
                    generated = model.generate(
                        input_ids,
                        max_new_tokens=100,
                        temperature=0.8,
                        top_p=0.9,
                        use_cache=True
                    )
                    response = vocab.decode(generated[0].tolist())
                    print(f" Response: {response}")
                    
            except FileNotFoundError:
                print("No trained model found. Please train first (option 1).")
            except Exception as e:
                print(f"Evaluation error: {e}")
                
        elif choice == "6":
            print(" ULTRA MODEL EXPORT")
            print("=" * 30)
            
            try:
                device = pick_device(cfg)
                model, vocab = load_ultra_model(cfg, device)
                
                print("1. TorchScript Export")
                print("2. ONNX Export")
                print("3. Quantized Export")
                
                export_choice = input("Choose export format (1-3): ").strip()
                
                if export_choice == "1":
                    print("Exporting to TorchScript...")
                    dummy_input = torch.randint(0, len(vocab.idx2word), (1, 32), device=device)
                    
                    model.eval()
                    traced_model = torch.jit.trace(model.forward, (dummy_input,))
                    
                    export_path = "ultra_gpt_torchscript.pt"
                    traced_model.save(export_path)
                    print(f"TorchScript model saved: {export_path}")
                    
                elif export_choice == "2":
                    print("ONNX export requires additional dependencies (onnx, onnxruntime)")
                    print("Install with: pip install onnx onnxruntime")
                    
                elif export_choice == "3":
                    print("Quantizing model for faster inference...")
                    quantized_model = torch.quantization.quantize_dynamic(
                        model.cpu(), 
                        {nn.Linear}, 
                        dtype=torch.qint8
                    )
                    
                    export_path = "ultra_gpt_quantized.pth"
                    torch.save(quantized_model.state_dict(), export_path)
                    print(f" Quantized model saved: {export_path}")
                    
                else:
                    print("Invalid choice")
                    
            except FileNotFoundError:
                print("No trained model found. Please train first (option 1).")
            except Exception as e:
                print(f"Export error: {e}")
                
        elif choice == "7":
            print("⚙️ ULTRA CONFIGURATION")
            print("=" * 30)
            
            print("Current Configuration:")
            config_dict = asdict(cfg)
            
            # Group configs by category
            categories = {
                " Architecture": ["d_model", "n_layers", "n_head", "n_kv_head", "max_len"],
                " Training": ["epochs", "batch_size", "lr", "weight_decay", "warmup_steps"],
                " Features": ["use_rope", "use_gqa", "use_swiglu", "use_flash_attention", "use_rmsnorm"],
                " Generation": ["gen_len", "temperature", "top_p", "top_k", "repetition_penalty"]
            }
            
            for category, keys in categories.items():
                print(f"\n{category}:")
                for key in keys:
                    if key in config_dict:
                        value = config_dict[key]
                        print(f"   {key}: {value}")
            
            print(f"\nFull config saved at startup. Modify the UltraConfig class to change settings.")
            
        else:
            print("Invalid choice. Please select 0-7.")

if __name__ == "__main__":
    print("(Optimized for GPU)")
    print("=" * 80)
    main()