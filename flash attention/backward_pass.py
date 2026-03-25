import math
import sys

try:
    import numpy as np
    NUMPY = True
except ImportError:
    NUMPY = False

TORCH = False

if not NUMPY and not TORCH:
    sys.exit("Need at least numpy. pip install numpy")

BACKEND = "torch" if TORCH else "numpy"
print(f"[INFO] Using {BACKEND} backend\n")



# Helpers

def zeros(*shape):
    if TORCH: return torch.zeros(*shape, dtype=torch.float64)
    else:     return np.zeros(shape, dtype=np.float64)

def to64(x):
    if TORCH: return x.double()
    else:     return x.astype(np.float64)

def matmul(A, B):
    if TORCH: return torch.matmul(A, B)
    else:     return A @ B

def exp(x):
    if TORCH: return torch.exp(x)
    else:     return np.exp(x)

def clamp_inf(x):
    if TORCH: return torch.where(torch.isneginf(x), torch.zeros_like(x), x)
    else:     return np.where(np.isneginf(x), 0.0, x)

def maximum(a, b):
    if TORCH: return torch.maximum(a, b)
    else:     return np.maximum(a, b)

def sum_last(x):
    if TORCH: return x.sum(dim=-1)
    else:     return x.sum(axis=-1)

def swap_last2(x):
    if TORCH: return x.transpose(-2, -1)
    else:     return x.swapaxes(-2, -1)

def max_diff(a, b):
    if TORCH: return float((to64(a) - to64(b)).abs().max())
    else:     return float(abs(to64(a) - to64(b)).max())

def standard_attention_backward(Q, K, V, O, dO, P):
    d = Q.shape[-1]
    sc = 1.0 / math.sqrt(d)

    dV = matmul(swap_last2(P), dO)
    dP = matmul(dO, swap_last2(V))

    if TORCH:
        D = (P * dP).sum(dim=-1)
    else:
        D = (P * dP).sum(axis=-1)

    dS = P * (dP - D[..., None])

    dQ = matmul(dS, K) * sc
    dK = matmul(swap_last2(dS), Q) * sc

    return dQ, dK, dV

def flash_attention_backward(Q, K, V, O, dO, L,
                              BLOCK_M: int = 32,
                              BLOCK_N: int = 32,
                              causal:  bool = False):
    B, H, N, d = Q.shape
    sc = 1.0 / math.sqrt(d)

    dQ = zeros(B, H, N, d)
    dK = zeros(B, H, N, d)
    dV = zeros(B, H, N, d)

    if TORCH:
        D = (to64(dO) * to64(O)).sum(dim=-1)
    else:
        D = (to64(dO) * to64(O)).sum(axis=-1)

    Q64  = to64(Q);   K64  = to64(K);   V64  = to64(V)
    dO64 = to64(dO);  L64  = to64(L)

    for tile_j in range(0, N, BLOCK_N):
        je  = min(tile_j + BLOCK_N, N)
        Kj  = K64[:, :, tile_j:je, :]
        Vj  = V64[:, :, tile_j:je, :]
        j_idx = list(range(tile_j, je))

        dKj_acc = zeros(B, H, je - tile_j, d)
        dVj_acc = zeros(B, H, je - tile_j, d)

        for tile_i in range(0, N, BLOCK_M):
            ie   = min(tile_i + BLOCK_M, N)
            Qi   = Q64[:, :, tile_i:ie, :]
            Oi   = to64(O)[:, :, tile_i:ie, :]
            dOi  = dO64[:, :, tile_i:ie, :]
            Li   = L64[:, :, tile_i:ie]
            Di   = D[:, :, tile_i:ie]
            i_idx = list(range(tile_i, ie))

            Sij = matmul(Qi, swap_last2(Kj)) * sc

            if causal:
                for bi in range(len(i_idx)):
                    for bj in range(len(j_idx)):
                        if i_idx[bi] < j_idx[bj]:
                            if TORCH: Sij[:,:,bi,bj] = float('-inf')
                            else:     Sij[:,:,bi,bj] = -np.inf

            Pij = exp(Sij - Li[..., None])
            if TORCH:
                Pij = torch.where(torch.isnan(Pij), torch.zeros_like(Pij), Pij)
            else:
                Pij = np.where(np.isnan(Pij), 0.0, Pij)

            dVj_acc += matmul(swap_last2(Pij), dOi)
            dPij = matmul(dOi, swap_last2(Vj))
            dSij = Pij * (dPij - Di[..., None])

            if TORCH:
                dQ[:, :, tile_i:ie, :] += (matmul(dSij, Kj) * sc).float()
            else:
                dQ[:, :, tile_i:ie, :] += matmul(dSij, Kj) * sc

            dKj_acc += matmul(swap_last2(dSij), Qi) * sc

        if TORCH:
            dK[:, :, tile_j:je, :] += dKj_acc.float()
            dV[:, :, tile_j:je, :] += dVj_acc.float()
        else:
            dK[:, :, tile_j:je, :] += dKj_acc
            dV[:, :, tile_j:je, :] += dVj_acc

    if TORCH:
        return dQ.float(), dK.float(), dV.float()
    else:
        return dQ, dK, dV

def compute_attention_and_P(Q, K, V, causal=False):
    d  = Q.shape[-1]
    sc = 1.0 / math.sqrt(d)
    if TORCH:
        S = matmul(Q, swap_last2(K)) * sc
        if causal:
            N = Q.shape[-2]
            mask = torch.triu(torch.ones(N,N,dtype=torch.bool,device=Q.device), 1)
            S = S.masked_fill(mask, float('-inf'))
        P = torch.softmax(S.float(), dim=-1)
        O = matmul(P, V.float())
        L = torch.logsumexp(S.float(), dim=-1)
        return O.to(Q.dtype), P, L
    else:
        S = to64(Q) @ to64(K).swapaxes(-2,-1) * sc
        if causal:
            N = Q.shape[-2]
            mask = np.triu(np.ones((N,N),bool),k=1)
            S[:,:,mask] = -np.inf
        Sm = S - S.max(axis=-1, keepdims=True)
        eS = np.exp(Sm)
        P  = eS / eS.sum(axis=-1, keepdims=True)
        O  = P @ to64(V)
        L  = np.log(eS.sum(axis=-1)) + S.max(axis=-1)
        return O.astype(Q.dtype), P, L

def randn(*shape, seed=0):
    if TORCH:
        torch.manual_seed(seed)
        return torch.randn(*shape)
    else:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape).astype(np.float32)

def run_tests():
    print("=" * 65)
    print("SECTION 3: FlashAttention Backward Pass — Verification")
    print("=" * 65)

    print("\n[A] Gradient correctness: flash_bwd == standard_bwd")
    print(f"    {'config':<35} {'dQ err':>8} {'dK err':>8} {'dV err':>8}")
    print(f"    {'-'*65}")

    configs = [
        (1, 1, 16,  16, 4,  4,  False, "B=1 H=1 N=16  d=16"),
        (1, 1, 64,  32, 16, 16, False, "B=1 H=1 N=64  d=32"),
        (2, 4, 128, 64, 32, 32, False, "B=2 H=4 N=128 d=64"),
        (1, 2, 64,  32, 16, 16, True,  "B=1 H=2 N=64  d=32 causal"),
        (2, 2, 32,  16, 8,  8,  True,  "B=2 H=2 N=32  d=16 causal"),
    ]

    all_pass = True
    for B,H,N,d,bm,bn,causal,label in configs:
        Q  = randn(B,H,N,d, seed=1)
        K  = randn(B,H,N,d, seed=2)
        V  = randn(B,H,N,d, seed=3)
        dO = randn(B,H,N,d, seed=4)

        O, P, L = compute_attention_and_P(Q, K, V, causal)

        # Reference: standard backward (needs P)
        dQ_ref, dK_ref, dV_ref = standard_attention_backward(Q, K, V, O, dO, P)

        # FlashAttention backward (uses only O and L)
        dQ_fa, dK_fa, dV_fa   = flash_attention_backward(
            Q, K, V, O, dO, L, bm, bn, causal)

        eq = max_diff(dQ_ref, dQ_fa)
        ek = max_diff(dK_ref, dK_fa)
        ev = max_diff(dV_ref, dV_fa)
        ok = max(eq, ek, ev) < 1e-4
        all_pass = all_pass and ok

        status = "PASS" if ok else "FAIL"
        print(f"    {status}  {label:<35} {eq:>8.1e} {ek:>8.1e} {ev:>8.1e}")

    print("\n[B] D_i trick: rowsum(P⊙dP) == (dO)^T O  (Eq. 4 in paper)")
    print(f"    {'config':<30} {'max_err':>10} {'status'}")
    print(f"    {'-'*50}")

    for B,H,N,d in [(1,1,32,16),(2,4,64,32)]:
        Q  = randn(B,H,N,d, seed=5)
        K  = randn(B,H,N,d, seed=6)
        V  = randn(B,H,N,d, seed=7)
        dO = randn(B,H,N,d, seed=8)
        O, P, L = compute_attention_and_P(Q, K, V)
        dP = matmul(dO, swap_last2(V))

        if TORCH:
            D_standard = (to64(P) * to64(dP)).sum(dim=-1)
            D_flash    = (to64(dO) * to64(O)).sum(dim=-1)
        else:
            D_standard = (to64(P) * to64(dP)).sum(axis=-1)
            D_flash    = (to64(dO) * to64(O)).sum(axis=-1)

        err = max_diff(D_standard, D_flash)
        ok  = err < 1e-5   # float32 inputs → ~1e-7 round-trip error is normal
        print(f"    {'PASS' if ok else 'FAIL'}  B={B} H={H} N={N} d={d}              err={err:.2e}")
        all_pass = all_pass and ok

    print("\n[C] Memory saved: not storing N×N P during backward")
    print(f"    {'N':>8} {'Store P (MB)':>14} {'Store L (KB)':>14} {'Saving':>10}")
    print(f"    {'-'*52}")
    for N in [512, 1024, 4096, 16384]:
        p_mb = N*N*2/1e6   # float16
        l_kb = N*4/1e3     # float32
        print(f"    {N:>8} {p_mb:>14.2f} {l_kb:>14.2f} {p_mb*1e3/l_kb:>9.0f}×")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()