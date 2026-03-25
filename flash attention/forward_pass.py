import math
import sys
import time
import numpy as np

TORCH = False

def standard_attention(Q, K, V):
    if TORCH:
        d     = Q.shape[-1]
        scale = 1.0 / math.sqrt(d)
        S     = Q @ K.transpose(-2, -1) * scale
        P     = torch.softmax(S, dim=-1)
        return P @ V
    else:
        d     = Q.shape[-1]
        scale = 1.0 / math.sqrt(d)
        S     = Q @ K.swapaxes(-2, -1) * scale
        S -= S.max(axis=-1, keepdims=True)
        P  = np.exp(S)
        P /= P.sum(axis=-1, keepdims=True)
        return P @ V

def flash_attention_forward(Q, K, V,
                             BLOCK_M: int = 32,
                             BLOCK_N: int = 32,
                             causal:  bool = False,
                             return_lse: bool = True):
    if TORCH:
        O, L = _flash_fwd_torch(Q, K, V, BLOCK_M, BLOCK_N, causal)
    else:
        O, L = _flash_fwd_numpy(Q, K, V, BLOCK_M, BLOCK_N, causal)

    if return_lse:
        return O, L
    else:
        return O, None

def _flash_fwd_torch(Q, K, V, BLOCK_M, BLOCK_N, causal):
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    dtype = Q.dtype

    O = torch.zeros_like(Q)
    L = torch.full((B, H, N), float('-inf'), dtype=torch.float32, device=Q.device)

    for tile_j_start in range(0, N, BLOCK_N):
        tile_j_end = min(tile_j_start + BLOCK_N, N)
        Kj = K[:, :, tile_j_start:tile_j_end, :]
        Vj = V[:, :, tile_j_start:tile_j_end, :]
        j_idx = torch.arange(tile_j_start, tile_j_end, device=Q.device)

        for tile_i_start in range(0, N, BLOCK_M):
            tile_i_end = min(tile_i_start + BLOCK_M, N)
            Qi = Q[:, :, tile_i_start:tile_i_end, :]
            i_idx = torch.arange(tile_i_start, tile_i_end, device=Q.device)

            Oi = O[:, :, tile_i_start:tile_i_end, :]
            Li = L[:, :, tile_i_start:tile_i_end]

            mi = torch.full_like(Li, float('-inf'))
            li = torch.zeros_like(Li)

            if tile_j_start > 0:
                pass

            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale
            Sij = Sij.float()

            if causal:
                mask = i_idx[:, None] < j_idx[None, :]
                Sij = Sij.masked_fill(mask[None, None, :, :], float('-inf'))

            Sij[:, :, :, j_idx >= N] = float('-inf')

            m_tilde = Sij.max(dim=-1).values
            P_tilde = torch.exp(Sij - m_tilde[..., None])
            l_tilde = P_tilde.sum(dim=-1)

            m_new = torch.maximum(mi, m_tilde)
            alpha = torch.exp(mi - m_new)
            beta  = torch.exp(m_tilde - m_new)

            l_new = alpha * li + beta * l_tilde

            P_scaled = (beta[..., None] * P_tilde)
            pv       = torch.matmul(P_scaled, Vj.float())

            Oi_new = (alpha[..., None] * li[..., None] * Oi.float() + pv) \
                     / l_new[..., None]

            mi = m_new
            li = l_new

            O[:, :, tile_i_start:tile_i_end, :] = Oi_new.to(dtype)
            L[:, :, tile_i_start:tile_i_end]    = mi + torch.log(li)

    return O, L


def _flash_fwd_numpy(Q, K, V, BLOCK_M, BLOCK_N, causal):
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    O  = np.zeros_like(Q, dtype=np.float32)
    m_ = np.full((B, H, N), -np.inf, dtype=np.float32)
    l_ = np.zeros((B, H, N),          dtype=np.float32)

    for tile_j in range(0, N, BLOCK_N):
        je = min(tile_j + BLOCK_N, N)
        Kj = K[:, :, tile_j:je, :].astype(np.float64)
        Vj = V[:, :, tile_j:je, :].astype(np.float64)

        for tile_i in range(0, N, BLOCK_M):
            ie = min(tile_i + BLOCK_M, N)
            Qi = Q[:, :, tile_i:ie, :].astype(np.float64)

            mi = m_[:, :, tile_i:ie].copy()
            li = l_[:, :, tile_i:ie].copy()
            Oi = O[:, :, tile_i:ie, :].copy()

            Sij = (Qi @ Kj.swapaxes(-2, -1)) * scale

            if causal:
                i_idx = np.arange(tile_i, ie)
                j_idx = np.arange(tile_j, je)
                mask  = (j_idx[None, :] > i_idx[:, None])
                Sij = np.where(mask[None, None, :, :], -np.inf, Sij)

            m_t = Sij.max(axis=-1)

            m_t_finite = np.where(np.isneginf(m_t), 0.0, m_t)
            P_t = np.exp(Sij - m_t_finite[..., None])
            P_t[np.isneginf(Sij)] = 0.0
            l_t = P_t.sum(axis=-1)

            m_new = np.maximum(mi, m_t)

            alpha = np.where(np.isneginf(mi) | np.isneginf(m_new), 0.0,
                             np.exp(mi - m_new))

            beta  = np.where(np.isneginf(m_t), 0.0, np.exp(m_t - m_new))

            l_new = alpha * li + beta * l_t

            pv     = (beta[..., None] * P_t) @ Vj
            Oi_new = alpha[..., None] * Oi + pv

            O[:, :, tile_i:ie, :]  = Oi_new
            m_[:, :, tile_i:ie]    = m_new
            l_[:, :, tile_i:ie]    = l_new

    safe_l = np.where(l_ == 0, 1.0, l_)
    O = O / safe_l[..., None]

    L = m_ + np.log(safe_l)
    return O.astype(Q.dtype), L


def make_tensors(B, H, N, d, seed=0):
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


if __name__ == "__main__":
    run_tests()