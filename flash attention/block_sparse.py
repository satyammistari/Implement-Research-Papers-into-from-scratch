

import math, sys, random
import numpy as np

# Force numpy backend
TORCH = False

# try:
#     import torch
#     _ = torch.zeros(1)  # Test if torch actually works
#     TORCH = True
# except (ImportError, OSError, RuntimeError, TypeError):
#     TORCH = False

print(f"[INFO] Using {'torch' if TORCH else 'numpy'} backend\n")



# 4A  Sparsity mask generators


def causal_mask(N, Br, Bc):
    
    Tr, Tc = math.ceil(N/Br), math.ceil(N/Bc)
    return [[1 if i*Br >= j*Bc else 0 for j in range(Tc)] for i in range(Tr)]

def local_window_mask(N, Br, Bc, window=2):
    
    Tr, Tc = math.ceil(N/Br), math.ceil(N/Bc)
    return [[1 if abs(i-j) <= window else 0 for j in range(Tc)] for i in range(Tr)]

def butterfly_mask(N, Br, Bc):
    
    Tr, Tc = math.ceil(N/Br), math.ceil(N/Bc)
    mask = [[0]*Tc for _ in range(Tr)]
    for i in range(min(Tr, Tc)):
        mask[i][i] = 1
    stride = 1
    while stride < max(Tr, Tc):
        for i in range(Tr):
            j = i ^ stride
            if 0 <= j < Tc:
                mask[i][j] = 1
        stride <<= 1
    return mask

def random_block_mask(Tr, Tc, density=0.3, seed=0):
    rng = random.Random(seed)
    return [[1 if rng.random() < density else 0 for _ in range(Tc)] for _ in range(Tr)]
def biology_informed_mask(N: int, Br: int, Bc: int) -> list[list[list[int]]]:

    Tr, Tc = math.ceil(N/Br), math.ceil(N/Bc)
    mask = [[0]*Tc for _ in range(Tr)]

    for i in range(Tr):
        for j in range(Tc):
            dist = abs(i - j) * Br
        
        #always attend to self and immediate neighbours
        if dist == 0:
            mask[i][j] = 1
        
        #local chromatin - short range interactions
        elif dist < 1_000:
            mask[i][j] = 1

        #enhance -promoter loops
        elif 40_000 <= dist < 60_000:
            mask[i][j] = 1

        #TAD boundaries
        elif (dist % 1_000_000) < 10_000:
            mask[i][j] = 1

    return mask    


def bilogy_maskl_stats(mask: list[list[int]]) -> dict:

    Tr, Tc = len(mask), len(mask[0])
    total_blocks = Tr * Tc
    nonzero_blocks = sum(mask[i][j] for i in range(Tr) for j in range(Tc)) 

    return{
        "total_blocks": total_blocks,
        "nonzero_blocks": nonzero_blocks,
        "sparsity": nonzero_blocks / total_blocks,
        "skip_fractions": 1 - nonzero_blocks / total_blocks,
        "io_reduction": f"{(1 / nonzero_blocks / total_blocks):.1f}x vs dense"
    }
    





def sparsity(mask):
    Tr, Tc = len(mask), len(mask[0])
    return sum(mask[i][j] for i in range(Tr) for j in range(Tc)) / (Tr * Tc)


# 4B  One-tile online softmax update  (identical recipe to Section 2)

def _update(Oi, mi, li, Sij, Vj):
    
    m_t      = Sij.max(axis=-1)
    m_t_safe = np.where(np.isneginf(m_t), 0.0, m_t)
    P_t      = np.exp(Sij - m_t_safe[..., None])
    P_t      = np.where(np.isneginf(Sij), 0.0, P_t)
    l_t      = P_t.sum(axis=-1)

    m_new = np.maximum(mi, m_t_safe)
    alpha = np.where(np.isneginf(mi),  0.0, np.exp(mi - m_new))
    beta  = np.where(np.isneginf(m_t), 0.0,
                     np.exp(np.where(np.isneginf(m_t), 0.0, m_t_safe - m_new)))

    l_new = alpha * li + beta * l_t
    pv    = (beta[..., None] * P_t) @ Vj
    safe  = np.where(l_new == 0, 1.0, l_new)
    O_new = (alpha[..., None] * li[..., None] * Oi + pv) / safe[..., None]
    return O_new, m_new, l_new


# 4C  Dense reference with block mask

def masked_attention_reference(Q, K, V, block_mask, Br, Bc):
    
    B, H, N, d = Q.shape
    sc  = 1.0 / math.sqrt(d)
    Tr  = math.ceil(N / Br); Tc = math.ceil(N / Bc)
    Q64 = Q.astype(np.float64)
    K64 = K.astype(np.float64)
    V64 = V.astype(np.float64)
    S   = np.full((B, H, N, N), -np.inf, dtype=np.float64)
    for i in range(Tr):
        for j in range(Tc):
            if block_mask[i][j]:
                ir = slice(i*Br, min((i+1)*Br, N))
                jr = slice(j*Bc, min((j+1)*Bc, N))
                S[:, :, ir, jr] = (Q64[:, :, ir, :] @
                                   K64[:, :, jr, :].swapaxes(-2, -1)) * sc
    Sm  = np.where(np.isneginf(S), S, S - S.max(axis=-1, keepdims=True))
    eS  = np.where(np.isneginf(Sm), 0.0, np.exp(Sm))
    den = eS.sum(axis=-1, keepdims=True)
    den = np.where(den == 0, 1.0, den)
    return (eS / den @ V64).astype(Q.dtype)


# 4D  Block-Sparse FlashAttention forward  (Algorithm 5)

def block_sparse_flash_attention(Q, K, V, block_mask, BLOCK_M=32, BLOCK_N=32):
    
    B, H, N, d = Q.shape
    sc  = 1.0 / math.sqrt(d)
    Tr  = math.ceil(N / BLOCK_M)
    Tc  = math.ceil(N / BLOCK_N)

    Q64 = Q.astype(np.float64)
    K64 = K.astype(np.float64)
    V64 = V.astype(np.float64)

    # Running state — lives in HBM here, but in Triton kernel stays in registers
    m_ = np.full( (B, H, N),    -np.inf, dtype=np.float64)
    l_ = np.zeros((B, H, N),             dtype=np.float64)
    O_ = np.zeros((B, H, N, d),          dtype=np.float64)

    blocks_computed = 0

    #  Outer loop: K/V tiles (j = 1..Tc)
    for tile_j in range(Tc):
        j_s = tile_j * BLOCK_N
        j_e = min(j_s + BLOCK_N, N)
        Kj  = K64[:, :, j_s:j_e, :]      
        Vj  = V64[:, :, j_s:j_e, :]      

        #  Inner loop: Q tiles (i = 1..Tr)
        for tile_i in range(Tr):

            if block_mask[tile_i][tile_j] == 0:
                continue

            blocks_computed += 1
            i_s = tile_i * BLOCK_M
            i_e = min(i_s + BLOCK_M, N)
            Qi  = Q64[:, :, i_s:i_e, :]

            # Load running state (read from HBM; in Triton: already in registers)
            mi = m_[:, :, i_s:i_e].copy()
            li = l_[:, :, i_s:i_e].copy()
            Oi = O_[:, :, i_s:i_e, :].copy()

            # Score tile — stays in SRAM/registers, never written to HBM
            Sij = (Qi @ Kj.swapaxes(-2, -1)) * sc

            # Online softmax update (Algorithm 1 lines 10-13)
            O_new, m_new, l_new = _update(Oi, mi, li, Sij, Vj)

            # Write back running state
            O_[:, :, i_s:i_e, :] = O_new
            m_[:, :, i_s:i_e]    = m_new
            l_[:, :, i_s:i_e]    = l_new

    safe_l = np.where(l_ == 0, 1.0, l_)
    L = m_ + np.log(safe_l)
    return O_.astype(Q.dtype), L, blocks_computed



# 4E  Verification + analysis

def randn(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def run_tests():
    print("=" * 65)
    print(" Block-Sparse FlashAttention — Verification")
    print("=" * 65)

    BM, BN = 16, 16
    B, H, N, d = 1, 2, 64, 32
    Q = randn(B,H,N,d,seed=1); K = randn(B,H,N,d,seed=2); V = randn(B,H,N,d,seed=3)
    Tr = math.ceil(N/BM); Tc = math.ceil(N/BN)

    #  A: correctness across all mask types 
    print("\n[A] Output: block_sparse_FA == masked_reference  (tol=1e-4)")
    print(f"    {'mask type':<28} {'sparsity':>9} {'max_err':>10} {'blocks':>8}")
    print(f"    {'-'*62}")

    masks = {
        "dense (all ones)":   [[1]*Tc for _ in range(Tr)],
        "causal":             causal_mask(N, BM, BN),
        "local window ±2":    local_window_mask(N, BM, BN, window=2),
        "butterfly":          butterfly_mask(N, BM, BN),
        "random 40%":         random_block_mask(Tr, Tc, density=0.4, seed=7),
        "random 10%":         random_block_mask(Tr, Tc, density=0.1, seed=8),
    }

    all_pass = True
    for name, mask in masks.items():
        s   = sparsity(mask)
        ref = masked_attention_reference(Q, K, V, mask, BM, BN)
        got, L_got, nb = block_sparse_flash_attention(Q, K, V, mask, BM, BN)
        err = float(abs(ref.astype(np.float64) - got.astype(np.float64)).max())
        ok  = err < 1e-4
        all_pass = all_pass and ok
        print(f"    {'PASS' if ok else 'FAIL'}  {name:<28} {s:>9.3f} {err:>10.2e} {nb:>8}")

    #  B: logsumexp L is correct 
    print("\n[B] Logsumexp L correctness (dense mask)")
    sc_ = 1.0 / math.sqrt(d)
    S   = (Q.astype(np.float64) @ K.astype(np.float64).swapaxes(-2,-1)) * sc_
    L_ref = np.log(np.exp(S - S.max(axis=-1,keepdims=True)).sum(axis=-1)) + S.max(axis=-1)
    dense_mask = [[1]*Tc for _ in range(Tr)]
    _, L_got, _ = block_sparse_flash_attention(Q, K, V, dense_mask, BM, BN)
    err_L = float(abs(L_ref - L_got).max())
    ok_L  = err_L < 1e-4
    all_pass = all_pass and ok_L
    print(f"    {'PASS' if ok_L else 'FAIL'}  max_err={err_L:.2e}")

    #  C: IO savings table 
    print("\n[C] Block-compute fraction  (blocks executed / total blocks)")
    print(f"    {'mask type':<22} {'N=512':>8} {'N=2048':>8} {'N=8192':>8}")
    print(f"    {'-'*50}")
    BM2, BN2 = 32, 32
    builders = {
        "causal":           lambda N,R,C: causal_mask(N,R,C),
        "local window ±3":  lambda N,R,C: local_window_mask(N,R,C,window=3),
        "butterfly":        lambda N,R,C: butterfly_mask(N,R,C),
        "random 10%":       lambda N,R,C: random_block_mask(
                                math.ceil(N/R),math.ceil(N/C),0.1),
    }
    for name, builder in builders.items():
        row = f"    {name:<22}"
        for N2 in [512, 2048, 8192]:
            row += f" {sparsity(builder(N2,BM2,BN2)):>8.3f}"
        print(row)

    #  D: IO complexity numbers 
    print("\n[D] Proposition 4 IO bytes (float16, d=64, M=100KB)")
    d3, M3 = 64, 100_000; Bc3 = M3//(4*d3)
    print(f"    {'N':>8} {'Standard':>14} {'Dense FA':>12} {'Causal FA':>12} {'Butterfly':>12}")
    print(f"    {'-'*62}")
    for N3 in [1024, 4096, 16384, 65536]:
        s_c = sparsity(causal_mask(N3, 64, 64))
        s_b = sparsity(butterfly_mask(N3, 64, 64))
        std = (N3*N3 + N3*d3)*2
        fa  = (N3*d3 + 2*N3*d3*(N3//Bc3) + N3*d3)*2
        print(f"    {N3:>8} {std/1e6:>13.1f}M {fa/1e6:>11.2f}M "
              f"{fa*s_c/1e6:>11.2f}M {fa*s_b/1e6:>11.2f}M")

    print("\n" + "="*65)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("="*65)
    

if __name__ == "__main__":
    run_tests()