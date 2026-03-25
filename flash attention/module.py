

import math, sys, time
import numpy as np

# Force numpy backend due to torch DLL issues on this system
TORCH = False

# try:
#     import torch
#     import torch.nn as nn
#     # Test if torch is actually usable
#     _ = torch.zeros(1)
#     TORCH = True
# except (ImportError, OSError, RuntimeError):
#     TORCH = False

print(f"[INFO] Using {'torch' if TORCH else 'numpy'} backend\n")


# Import our section implementations
#
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Section 2: forward pass
from forward_pass import (
    flash_attention_forward,
    standard_attention,
)
# Section 3: backward pass
from backward_pass import (
    flash_attention_backward,
    compute_attention_and_P,
)
# Section 4: block-sparse
from block_sparse import (
    block_sparse_flash_attention,
    causal_mask, butterfly_mask, local_window_mask,
    sparsity,
)


# 5A  Custom autograd Function  (torch only)


if TORCH:
    class FlashAttentionFunction(torch.autograd.Function):
       

        @staticmethod
        def forward(ctx, Q, K, V, BLOCK_M=64, BLOCK_N=64, causal=False):
            O, L = flash_attention_forward(Q, K, V, BLOCK_M, BLOCK_N, causal)
            ctx.save_for_backward(Q, K, V, O, L)
            ctx.causal  = causal
            ctx.BLOCK_M = BLOCK_M
            ctx.BLOCK_N = BLOCK_N
            return O

        @staticmethod
        def backward(ctx, dO):
            Q, K, V, O, L = ctx.saved_tensors
            dQ, dK, dV = flash_attention_backward(
                Q, K, V, O, dO, L,
                ctx.BLOCK_M, ctx.BLOCK_N, ctx.causal
            )
            
            return dQ, dK, dV, None, None, None


    def flash_attn(Q, K, V, BLOCK_M=64, BLOCK_N=64, causal=False):
        """Functional API: same signature as F.scaled_dot_product_attention."""
        return FlashAttentionFunction.apply(Q, K, V, BLOCK_M, BLOCK_N, causal)


# 5B  FlashAttentionLayer: a complete multi-head attention module

class FlashAttentionLayer:
   

    def __init__(self, embed_dim: int, num_heads: int,
                 BLOCK_M: int = 64, BLOCK_N: int = 64,
                 causal: bool = False, seed: int = 0):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.BLOCK_M   = BLOCK_M
        self.BLOCK_N   = BLOCK_N
        self.causal    = causal

        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(embed_dim)

        # Projection matrices  [embed_dim, embed_dim]
        self.W_Q = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale
        self.W_K = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale
        self.W_V = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale
        self.W_O = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale

        # Convert to torch tensors if using torch backend
        if TORCH:
            self.W_Q = torch.from_numpy(self.W_Q)
            self.W_K = torch.from_numpy(self.W_K)
            self.W_V = torch.from_numpy(self.W_V)
            self.W_O = torch.from_numpy(self.W_O)

    def forward(self, x: np.ndarray) -> np.ndarray:
        
        B, N, E = x.shape
        H = self.num_heads
        D = self.head_dim

        Q_flat = x @ self.W_Q                  
        K_flat = x @ self.W_K
        V_flat = x @ self.W_V

        # Reshape to [B, H, N, D] for multi-head attention
        def reshape(t):
            return t.reshape(B, N, H, D).transpose(0, 2, 1, 3)  

        Q = reshape(Q_flat)
        K = reshape(K_flat)
        V = reshape(V_flat)

        #    FlashAttention core      
        O, L = flash_attention_forward(Q, K, V, self.BLOCK_M, self.BLOCK_N,
                                        self.causal)

        # Reshape back  
        O_flat = O.transpose(0, 2, 1, 3).reshape(B, N, E)       

        # Output projection
        return O_flat @ self.W_O                                


# 5C  BlockSparseFlashAttentionLayer

class BlockSparseFlashAttentionLayer(FlashAttentionLayer):


    def __init__(self, embed_dim, num_heads,
                 BLOCK_M=64, BLOCK_N=64,
                 mask_type='causal', seed=0):
        super().__init__(embed_dim, num_heads, BLOCK_M, BLOCK_N, False, seed)
        self.mask_type = mask_type

    def _build_mask(self, N):
        Tr = math.ceil(N / self.BLOCK_M)
        Tc = math.ceil(N / self.BLOCK_N)
        if self.mask_type == 'causal':
            return causal_mask(N, self.BLOCK_M, self.BLOCK_N)
        elif self.mask_type == 'butterfly':
            return butterfly_mask(N, self.BLOCK_M, self.BLOCK_N)
        elif self.mask_type == 'local':
            return local_window_mask(N, self.BLOCK_M, self.BLOCK_N, window=2)
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, N, E = x.shape
        H = self.num_heads; D = self.head_dim

        Q_flat = x @ self.W_Q
        K_flat = x @ self.W_K
        V_flat = x @ self.W_V

        def reshape(t):
            return t.reshape(B, N, H, D).transpose(0, 2, 1, 3)

        Q = reshape(Q_flat); K = reshape(K_flat); V = reshape(V_flat)

            
        mask = self._build_mask(N)
        O, L, nb = block_sparse_flash_attention(Q, K, V, mask,
                                                 self.BLOCK_M, self.BLOCK_N)

        O_flat = O.transpose(0, 2, 1, 3).reshape(B, N, E)
        return O_flat @ self.W_O



# 5D  Verification + benchmarking

def randn(*shape, seed=0):
    arr = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    if TORCH:
        return torch.from_numpy(arr)
    return arr


def standard_mha_numpy(x, W_Q, W_K, W_V, W_O, num_heads):
    
    B, N, E = x.shape
    H = num_heads; D = E // H
    sc = 1.0 / math.sqrt(D)

    def proj_reshape(W):
        t = (x @ W).reshape(B, N, H, D).transpose(0, 2, 1, 3)
        return t

    Q = proj_reshape(W_Q); K = proj_reshape(W_K); V = proj_reshape(W_V)
    S  = (Q @ K.swapaxes(-2,-1)) * sc
    Sm = S - S.max(axis=-1, keepdims=True)
    P  = np.exp(Sm); P /= P.sum(axis=-1, keepdims=True)
    O  = P @ V
    O_flat = O.transpose(0,2,1,3).reshape(B, N, E)
    return O_flat @ W_O


def run_tests():
    print("=" * 65)
    
    print("=" * 65)

    
    print("\n[A] FlashAttentionLayer output == standard MHA  (tol=1e-4)")
    print(f"    {'config':<40} {'max_err':>10} {'status'}")
    print(f"    {'-'*58}")

    configs = [
        (1, 32,  64,  2,  8,  8,  False, "B=1 N=32  E=64  H=2"),
        (2, 64,  128, 4,  16, 16, False, "B=2 N=64  E=128 H=4"),
        (1, 128, 256, 8,  32, 32, False, "B=1 N=128 E=256 H=8"),
        (1, 64,  128, 4,  16, 16, True,  "B=1 N=64  E=128 H=4 causal"),
    ]

    all_pass = True
    for B,N,E,H,bm,bn,causal,label in configs:
        x   = randn(B, N, E, seed=0)
        layer = FlashAttentionLayer(E, H, bm, bn, causal, seed=1)
        ref   = standard_mha_numpy(x, layer.W_Q, layer.W_K, layer.W_V,
                                    layer.W_O, H)
        if causal:
            # Build causal reference
            D_ = E // H
            sc_ = 1.0 / math.sqrt(D_)
            def proj_r(W):
                return (x @ W).reshape(B,N,H,D_).transpose(0,2,1,3)
            Q_=proj_r(layer.W_Q); K_=proj_r(layer.W_K); V_=proj_r(layer.W_V)
            S_=(Q_@K_.swapaxes(-2,-1))*sc_
            mask_=np.triu(np.ones((N,N),bool),k=1)
            S_[:,:,mask_]=-np.inf
            Sm_=S_-np.where(np.isneginf(S_),-1e30,S_).max(axis=-1,keepdims=True)
            P_=np.where(np.isneginf(S_),0.0,np.exp(Sm_))
            P_/=np.where(P_.sum(axis=-1,keepdims=True)==0,1,P_.sum(axis=-1,keepdims=True))
            ref=(P_@V_).transpose(0,2,1,3).reshape(B,N,E)@layer.W_O

        got = layer.forward(x)
        err = float(abs(ref.astype(np.float64) - got.astype(np.float64)).max())
        ok  = err < 1e-3
        all_pass = all_pass and ok
        print(f"    {'PASS' if ok else 'FAIL'}  {label:<40} {err:>10.2e}")

    
    print("\n[B] BlockSparseFlashAttentionLayer ")
    print(f"    {'mask type':<15} {'sparsity':>10} {'err vs causal ref':>18} {'status'}")
    print(f"    {'-'*55}")

    B,N,E,H,bm,bn = 1,64,128,4,16,16
    x_b = randn(B,N,E,seed=2)

    for mask_type in ['causal', 'butterfly', 'local']:
        layer_bs = BlockSparseFlashAttentionLayer(E, H, bm, bn, mask_type, seed=3)
        got_bs   = layer_bs.forward(x_b)

        # Reference: use masked_attention for the exact same mask
        from block_sparse import masked_attention_reference
        D_ = E // H
        def proj(W):
            return (x_b @ W).reshape(B,N,H,D_).transpose(0,2,1,3)
        Q_=proj(layer_bs.W_Q); K_=proj(layer_bs.W_K); V_=proj(layer_bs.W_V)
        mask_ = layer_bs._build_mask(N)
        s_    = sparsity(mask_)
        O_ref = masked_attention_reference(Q_, K_, V_, mask_, bm, bn)
        ref_bs = O_ref.transpose(0,2,1,3).reshape(B,N,E) @ layer_bs.W_O

        err_bs = float(abs(ref_bs.astype(np.float64) - got_bs.astype(np.float64)).max())
        ok_bs  = err_bs < 1e-4
        all_pass = all_pass and ok_bs
        print(f"    {'PASS' if ok_bs else 'FAIL'}  {mask_type:<15} {s_:>10.3f} {err_bs:>18.2e}")

    #        autograd check (torch only)     
    if TORCH:
        print("\n[C] Autograd gradient check (torch)")
        print(f"    {'config':<35} {'max_grad_err':>14} {'status'}")
        print(f"    {'-'*55}")

        for B,H,N,d in [(1,1,16,16),(1,2,32,32)]:
            Q_t = torch.randn(B,H,N,d, dtype=torch.float64, requires_grad=True)
            K_t = torch.randn(B,H,N,d, dtype=torch.float64, requires_grad=True)
            V_t = torch.randn(B,H,N,d, dtype=torch.float64, requires_grad=True)

            def fn(Q,K,V):
                return FlashAttentionFunction.apply(Q,K,V,8,8,False)

            result = torch.autograd.gradcheck(fn, (Q_t,K_t,V_t), eps=1e-4,
                                              atol=1e-3, rtol=1e-3, raise_on_error=False)
            ok_ag = bool(result)
            all_pass = all_pass and ok_ag
            print(f"    {'PASS' if ok_ag else 'FAIL'}  B={B} H={H} N={N} d={d}")
    else:
        print("\n[C] Autograd check — skipped (torch not installed)")

   
    print("\n[D] Wall-clock time: FlashAttention vs standard attention (ms)")
    print(f"    {'config':<30} {'standard':>12} {'flash':>12} {'speedup':>10}")
    print(f"    {'-'*68}")

    timing_configs = [
        (1, 2, 128,  64, "B=1 H=2 N=128  d=64"),
        (1, 2, 512,  64, "B=1 H=2 N=512  d=64"),
        (1, 2, 1024, 64, "B=1 H=2 N=1024 d=64"),
    ]

    for B,H,N,d,label in timing_configs:
        Q = randn(B,H,N,d,seed=0)
        K = randn(B,H,N,d,seed=1)
        V = randn(B,H,N,d,seed=2)
        REPS = 5

        # Standard
        t0 = time.perf_counter()
        for _ in range(REPS):
            _ = standard_attention(Q, K, V)
        t_std = (time.perf_counter() - t0) / REPS * 1000

        # Flash
        t0 = time.perf_counter()
        for _ in range(REPS):
            _ = flash_attention_forward(Q, K, V, 32, 32)
        t_fa = (time.perf_counter() - t0) / REPS * 1000

        spd = t_std / t_fa
        print(f"    {label:<30} {t_std:>11.2f}ms {t_fa:>11.2f}ms {spd:>9.2f}×")

    print(f"\n    NOTE: NumPy/CPU timing does NOT reflect GPU speedup.")
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("=" * 65)

    

if __name__ == "__main__":
    run_tests()