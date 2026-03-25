import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '.')
from forward_pass import flash_attention_forward, standard_attention
from block_sparse import (
    block_sparse_flash_attention,
    causal_mask, butterfly_mask, biology_informed_mask
)

VOCAB_SIZE = 7


class FlashAttentionHead(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 attn_type: str = 'flash_dense',
                 BLOCK_M: int = 32, BLOCK_N: int = 32,
                 budget: float = 0.2):

        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.attn_type  = attn_type
        self.BLOCK_M    = BLOCK_M
        self.BLOCK_N    = BLOCK_N
        self.budget     = budget

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        self.last_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, E = x.shape
        H, D    = self.num_heads, self.head_dim

        def reshape(t):
            return t.reshape(B, N, H, D).permute(0, 2, 1, 3)

        Q = reshape(self.W_Q(x))
        K = reshape(self.W_K(x))
        V = reshape(self.W_V(x))

        Q_np = Q.detach().cpu().float().numpy()
        K_np = K.detach().cpu().float().numpy()
        V_np = V.detach().cpu().float().numpy()

        if self.attn_type == 'standard':
            O_np = standard_attention(Q_np, K_np, V_np)

        elif self.attn_type == 'flash_dense':
            O_np, _ = flash_attention_forward(
                Q_np, K_np, V_np, self.BLOCK_M, self.BLOCK_N)

        elif self.attn_type == 'flash_biology':
            mask = biology_informed_mask(N, self.BLOCK_M, self.BLOCK_N)
            O_np, _, _ = block_sparse_flash_attention(
                Q_np, K_np, V_np, mask, self.BLOCK_M, self.BLOCK_N)

        elif self.attn_type == 'flash_butterfly':
            mask = butterfly_mask(N, self.BLOCK_M, self.BLOCK_N)
            O_np, _, _ = block_sparse_flash_attention(
                Q_np, K_np, V_np, mask, self.BLOCK_M, self.BLOCK_N)

        elif self.attn_type == 'flash_dynamic':
            import numpy as np
            Tr = math.ceil(N / self.BLOCK_M)
            Tc = math.ceil(N / self.BLOCK_N)

            Q_pool = Q_np.reshape(B,H,Tr,self.BLOCK_M,D).mean(axis=3)
            K_pool = K_np.reshape(B,H,Tc,self.BLOCK_N,D).mean(axis=3)
            scores = Q_pool @ K_pool.swapaxes(-2,-1)
            avg_sc = scores.mean(axis=(0,1))

            k    = max(1, int(self.budget * Tc))
            mask = [[0]*Tc for _ in range(Tr)]
            for i in range(Tr):
                for j in np.argpartition(avg_sc[i], -k)[-k:]:
                    mask[i][j] = 1

            self.last_mask = mask
            O_np, _, _ = block_sparse_flash_attention(
                Q_np, K_np, V_np, mask, self.BLOCK_M, self.BLOCK_N)

        O = torch.from_numpy(O_np).to(x.device).to(x.dtype)
        O = O.permute(0, 2, 1, 3).reshape(B, N, E)
        return self.W_O(O)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 attn_type: str, BLOCK_M: int, BLOCK_N: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = FlashAttentionHead(
            embed_dim, num_heads, attn_type, BLOCK_M, BLOCK_N)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyDNATransformer(nn.Module):
    def __init__(self, attn_type: str = 'flash_dense',
                 embed_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 3, max_len: int = 4096,
                 BLOCK_M: int = 32, BLOCK_N: int = 32):
        super().__init__()

        self.embed     = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, attn_type, BLOCK_M, BLOCK_N)
            for _ in range(num_layers)
        ])

        self.norm    = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, VOCAB_SIZE)

        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, N      = input_ids.shape
        positions = torch.arange(N, device=input_ids.device)
        x         = self.embed(input_ids) + self.pos_embed(positions)

        for block in self.blocks:
            x = block(x)

        return self.lm_head(self.norm(x))

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())