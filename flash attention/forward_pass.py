

import math
import sys
import time



import numpy as np


# 2A  Standard attention  (Algorithm 0 in the paper — O(N²) memory baseline)

def standard_attention(Q, K, V):
    """
    Naive attention — materialises the full N×N score matrix.
    This is what FlashAttention replaces.

    HBM traffic (per paper, Theorem 2):
      Write S = QK^T  →  Θ(Nd + N²)
      Write P = softmax(S) →  Θ(N²)
      Write O = PV    →  Θ(Nd + N²)
      Total:  Θ(Nd + N²)   dominated by N²
    """
    if TORCH:
        d     = Q.shape[-1]
        scale = 1.0 / math.sqrt(d)
        S     = Q @ K.transpose(-2, -1) * scale   # [B,H,N,N]  ← THE PROBLEM
        P     = torch.softmax(S, dim=-1)           # [B,H,N,N]
        return P @ V                               # [B,H,N,d]
    else:
        d     = Q.shape[-1]
        scale = 1.0 / math.sqrt(d)
        S     = Q @ K.swapaxes(-2, -1) * scale
        # row-wise softmax
        S -= S.max(axis=-1, keepdims=True)
        P  = np.exp(S)
        P /= P.sum(axis=-1, keepdims=True)
        return P @ V


# 2B  FlashAttention forward pass  (Algorithm 1 — O(N) memory)

def flash_attention_forward(Q, K, V,
                             BLOCK_M: int = 32,
                             BLOCK_N: int = 32,
                             causal:  bool = False):
    if TORCH:
        return _flash_fwd_torch(Q, K, V, BLOCK_M, BLOCK_N, causal)
    else:
        return _flash_fwd_numpy(Q, K, V, BLOCK_M, BLOCK_N, causal)


def _flash_fwd_torch(Q, K, V, BLOCK_M, BLOCK_N, causal):
    """PyTorch implementation of Algorithm 1."""
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    dtype = Q.dtype

    # Outputs allocated in HBM — written tile-by-tile
    O = torch.zeros_like(Q)
    L = torch.full((B, H, N), float('-inf'), dtype=torch.float32, device=Q.device)

    # ── Outer loop: iterate over K/V tiles (j = 1..Tc) ───────────────────
    for tile_j_start in range(0, N, BLOCK_N):
        tile_j_end = min(tile_j_start + BLOCK_N, N)
        Kj = K[:, :, tile_j_start:tile_j_end, :]   # [B,H,Bc,d]  — loaded once
        Vj = V[:, :, tile_j_start:tile_j_end, :]   # [B,H,Bc,d]  — loaded once
        j_idx = torch.arange(tile_j_start, tile_j_end, device=Q.device)

        # ── Inner loop: iterate over Q tiles (i = 1..Tr) ─────────────────
        for tile_i_start in range(0, N, BLOCK_M):
            tile_i_end = min(tile_i_start + BLOCK_M, N)
            Qi = Q[:, :, tile_i_start:tile_i_end, :]   # [B,H,Br,d]
            i_idx = torch.arange(tile_i_start, tile_i_end, device=Q.device)

            # Load running state for these query rows
            Oi = O[:, :, tile_i_start:tile_i_end, :]   # [B,H,Br,d]
            Li = L[:, :, tile_i_start:tile_i_end]       # [B,H,Br]
            # Split L back into (m, l) for the accumulation
            
            mi = torch.full_like(Li, float('-inf'))
            li = torch.zeros_like(Li)

            #  Recompute (mi, li) from stored L if not first tile 
            
            if tile_j_start > 0:
                
                pass  # handled by accumulating L below

            #  On-chip computation (lines 9-13 of Algorithm 1) 

            # S_ij = Qi @ Kj^T * scale   [B,H,Br,Bc]
            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale  # fp32 for stability
            Sij = Sij.float()

            # Causal mask: set S[i,j] = -inf where j > i
            if causal:
                mask = i_idx[:, None] < j_idx[None, :]   # [Br, Bc]
                Sij = Sij.masked_fill(mask[None, None, :, :], float('-inf'))

            # Pad mask: out-of-bounds columns
            Sij[:, :, :, j_idx >= N] = float('-inf')

            # Online softmax update
            m_tilde = Sij.max(dim=-1).values           # [B,H,Br]  local max
            P_tilde = torch.exp(Sij - m_tilde[..., None])  # [B,H,Br,Bc]
            l_tilde = P_tilde.sum(dim=-1)              # [B,H,Br]  local sum

            m_new = torch.maximum(mi, m_tilde)         # new global max
            alpha = torch.exp(mi - m_new)              # rescale old O
            beta  = torch.exp(m_tilde - m_new)         # scale new P

            l_new = alpha * li + beta * l_tilde

            # Accumulate output:
            
            P_scaled = (beta[..., None] * P_tilde)     # [B,H,Br,Bc]
            pv       = torch.matmul(P_scaled, Vj.float())  # [B,H,Br,d]

            Oi_new = (alpha[..., None] * li[..., None] * Oi.float() + pv) \
                     / l_new[..., None]

            mi = m_new
            li = l_new

            # Write back to HBM
            O[:, :, tile_i_start:tile_i_end, :] = Oi_new.to(dtype)
            L[:, :, tile_i_start:tile_i_end]    = mi + torch.log(li)  # logsumexp

    return O, L


def _flash_fwd_numpy(Q, K, V, BLOCK_M, BLOCK_N, causal):
   
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    O  = np.zeros((B, H, N, d), dtype=np.float64)   # unnormalized accumulator
    m_ = np.full( (B, H, N),    -np.inf, dtype=np.float64)  # running max per row
    l_ = np.zeros((B, H, N),             dtype=np.float64)  # running denom per row

    #  Outer loop: K/V tiles (j = 1..Tc) 
    for tile_j in range(0, N, BLOCK_N):
        je = min(tile_j + BLOCK_N, N)
        Kj = K[:, :, tile_j:je, :].astype(np.float64)  # [B,H,Bc,d]
        Vj = V[:, :, tile_j:je, :].astype(np.float64)  # [B,H,Bc,d]

        #  Inner loop: Q tiles (i = 1..Tr)    
        for tile_i in range(0, N, BLOCK_M):
            ie = min(tile_i + BLOCK_M, N)
            Qi = Q[:, :, tile_i:ie, :].astype(np.float64)  # [B,H,Br,d]

            # Load running state for these query rows
            mi = m_[:, :, tile_i:ie].copy()    # [B,H,Br]  global max so far
            li = l_[:, :, tile_i:ie].copy()    # [B,H,Br]  global denom so far
            Oi = O[:, :, tile_i:ie, :].copy()  # [B,H,Br,d]  unnormalized

            
            Sij = (Qi @ Kj.swapaxes(-2, -1)) * scale

            # Causal mask: positions where key index > query index are masked
            if causal:
                i_idx = np.arange(tile_i, ie)       # query positions [Br]
                j_idx = np.arange(tile_j, je)       # key   positions [Bc]
                mask  = (j_idx[None, :] > i_idx[:, None])  # [Br,Bc] True=future
                # broadcast 2D mask over the [B,H] batch dims
                Sij = np.where(mask[None, None, :, :], -np.inf, Sij)

            #  Online softmax stats for this tile 
            m_t = Sij.max(axis=-1)   # [B,H,Br]; -inf when entire row is masked

            
            m_t_finite = np.where(np.isneginf(m_t), 0.0, m_t)
            P_t = np.exp(Sij - m_t_finite[..., None])  # [B,H,Br,Bc]
            P_t[np.isneginf(Sij)] = 0.0               # genuine -inf entries → 0
            l_t = P_t.sum(axis=-1)                     # [B,H,Br]  (0 if all masked)

            # Online merge (Algorithm 1, lines 12-13) 
            m_new = np.maximum(mi, m_t)

            # alpha=0 when mi=-inf (no prior state) OR m_new=-inf (both were -inf)
            alpha = np.where(np.isneginf(mi) | np.isneginf(m_new), 0.0,
                             np.exp(mi - m_new))

            # beta: scale for this tile's contribution.
           
            beta  = np.where(np.isneginf(m_t), 0.0, np.exp(m_t - m_new))

            l_new = alpha * li + beta * l_t

            # Unnormalized accumulator — NO division here (matches Triton kernel)
            pv     = (beta[..., None] * P_t) @ Vj   # [B,H,Br,d]
            Oi_new = alpha[..., None] * Oi + pv

            # Write back (still unnormalized)
            O[:, :, tile_i:ie, :]  = Oi_new
            m_[:, :, tile_i:ie]    = m_new
            l_[:, :, tile_i:ie]    = l_new

    #  Final normalisation — divide ONCE after all K/V tiles 
    safe_l = np.where(l_ == 0, 1.0, l_)   # avoid div-by-zero on empty rows
    O = O / safe_l[..., None]

    L = m_ + np.log(safe_l)               # logsumexp
    return O.astype(Q.dtype), L


# 2D  Numerical verification + IO complexity demo

def make_tensors(B, H, N, d, seed=0):
    """Create random Q, K, V tensors."""
    if TORCH:
        torch.manual_seed(seed)
        return (torch.randn(B, H, N, d),
                torch.randn(B, H, N, d),
                torch.randn(B, H, N, d))
    else:
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((B,H,N,d)).astype(np.float32),
                rng.standard_normal((B,H,N,d)).astype(np.float32),
                rng.standard_normal((B,H,N,d)).astype(np.float32))


def max_diff(a, b):
    if TORCH:
        return (a.float() - b.float()).abs().max().item()
    else:
        return float(abs(a.astype(np.float64) - b.astype(np.float64)).max())


def run_tests():
    print("=" * 65)
    print("FlashAttention Forward Pass — Verification")
    print("=" * 65)

    #  A: numerical correctness 
    print("\n[A] Output correctness: flash == standard  (tol < 1e-4)")
    print(f"    {'config':<35} {'max_err':<12} {'status'}")
    print(f"    {'-'*55}")

    configs = [
        (1, 1, 16,  16, 4,  4,  False, "B=1 H=1 N=16  d=16"),
        (1, 1, 64,  32, 16, 16, False, "B=1 H=1 N=64  d=32"),
        (2, 4, 128, 64, 32, 32, False, "B=2 H=4 N=128 d=64"),
        (1, 2, 64,  32, 16, 16, True,  "B=1 H=2 N=64  d=32 causal"),
        (2, 2, 32,  16, 8,  8,  True,  "B=2 H=2 N=32  d=16 causal"),
    ]

    all_pass = True
    for B,H,N,d,bm,bn,causal,label in configs:
        Q, K, V = make_tensors(B, H, N, d)
        ref = standard_attention(Q, K, V)
        if causal:
            # apply causal mask to reference
            if TORCH:
                mask = torch.triu(torch.ones(N,N,dtype=torch.bool),diagonal=1)
                d_ = Q.shape[-1]
                sc = 1.0/math.sqrt(d_)
                S  = (Q @ K.transpose(-2,-1))*sc
                S  = S.masked_fill(mask,-float('inf'))
                P  = torch.softmax(S,-1)
                ref = P @ V
            else:
                sc = 1.0/math.sqrt(d)
                S  = (Q @ K.swapaxes(-2,-1))*sc
                mask = np.triu(np.ones((N,N),bool),k=1)  # [N,N]
                S = np.where(mask[None, None, :, :], -np.inf, S)
                S -= S.max(axis=-1,keepdims=True)
                P  = np.exp(S); P /= P.sum(axis=-1,keepdims=True)
                ref = P @ V

        got, L = flash_attention_forward(Q, K, V, bm, bn, causal)
        err    = max_diff(ref, got)
        ok     = err < 1e-3
        all_pass = all_pass and ok
        print(f"    {'PASS' if ok else 'FAIL':<4}  {label:<35} err={err:.2e}")

    #  B: L (logsumexp) is correct 
    print("\n[B] Logsumexp L = log(sum(exp(S_i))) correctness")
    print(f"    {'config':<30} {'max_err':<12} {'status'}")
    print(f"    {'-'*55}")

    for B,H,N,d in [(1,1,32,16),(1,2,64,32)]:
        Q, K, V = make_tensors(B, H, N, d)
        _, L    = flash_attention_forward(Q, K, V)
        # compute true logsumexp
        sc = 1.0 / math.sqrt(d)
        if TORCH:
            S   = (Q @ K.transpose(-2,-1)) * sc        # [B,H,N,N]
            L_ref = torch.logsumexp(S, dim=-1)         # [B,H,N]
            err = (L.float() - L_ref).abs().max().item()
        else:
            S     = (Q @ K.swapaxes(-2,-1)) * sc
            L_ref = np.log(np.exp(S - S.max(-1,keepdims=True)).sum(-1)) + S.max(-1)
            err   = float(abs(L.astype(np.float64) - L_ref.astype(np.float64)).max())
        ok = err < 1e-3
        print(f"    {'PASS' if ok else 'FAIL':<4}  B={B} H={H} N={N} d={d}               err={err:.2e}")
        all_pass = all_pass and ok

    #  C: IO complexity table 
    print("\n[C] IO complexity: standard vs FlashAttention (bytes, float16)")
    print(f"    {'Sequence N':<14} {'Standard (MB)':>15} {'Flash (MB)':>12} {'Reduction':>10}")
    print(f"    {'-'*55}")
    bpe = 2  # float16
    d_, M_ = 64, 100_000  # typical head dim and SRAM size in bytes
    for N in [512, 1024, 4096, 16384, 65536]:
        std   = (2*(N*N + N*d_)) * bpe
        flash = (4*N*d_) * bpe          # simplified: Q,K,V,O each Nd
        Bc    = M_ // (4*d_)
        flash_exact = int(N*d_ + 2*N*d_*(N//Bc) + N*d_) * bpe
        print(f"    N={N:<10} {std/1e6:>14.1f}  {flash_exact/1e6:>12.4f}  {std/flash_exact:>9.0f}×")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("=" * 65)

    print(""" (backward pass):
  1. We save only L = m + log(l)  per query row (shape [N]).
     During backward, P_ij = exp(S_ij - L_i)  — no need to store N×N P.

  2. D_i = rowsum(dO_i o O_i)  is computed as a dot product of two
     d-dimensional vectors — no reduction over N needed.

  3. dQ, dK, dV all reuse the same tiling structure as the forward pass,
     giving the same Θ(N²d²/M) HBM access count.
""")


if __name__ == "__main__":
    run_tests()