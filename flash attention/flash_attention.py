#online softmax- mathematics behind flash attention
import math

#naive softmax
def naive_softmax(x: list[float]) -> list[float]:
    m = max(x)
    exps = [math.exp(i - m) for i in x]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

#online softmax
def online_softmax(x: list[float], tile_size: int = 4) -> list[float]:
    N = len(x)
    m = -float('inf')

    l_sum = 0.0
    numerator = [0.0] * N  # fix: must be a list, not a float

    for start in range(0, N, tile_size):
        # ← "load tile from HBM"
        tile = x[start:start + tile_size]
        # ── local stats for this tile
        m_tile = max(tile)
        l_tile = sum(math.exp(i - m_tile) for i in tile)

        # online merge rule
        m_new = max(m, m_tile)
        l_new = l_sum * math.exp(m - m_new) + l_tile * math.exp(m_tile - m_new)

        # rescale old numerators to new global max
        for i in range(start):
            numerator[i] = numerator[i] * math.exp(m - m_new)

        # store numerators for this tile relative to new global max
        for i in range(len(tile)):  # fix: use len(tile) to handle last partial tile
            numerator[start + i] = math.exp(tile[i] - m_new)

        m = m_new
        l_sum = l_new

    # final normalization
    return [n / l_sum for n in numerator]  # fix: was 'numerators' (undefined)

# 1C  Streaming softmax with explicit O accumulator
def streaming_attention_row(
    q: list[float],
    K: list[list[float]],
    V: list[list[float]],
    tile_size: int = 4,
    scale: float | None = None,
) -> list[float]:

    N = len(K)
    d = len(q)
    if scale is None:
        scale = 1.0 / math.sqrt(d)
 
    # running state — lives in SRAM (registers) throughout
    m = float('-inf')
    l = 0.0
    o = [0.0] * d          # output accumulator
 
    for tile_start in range(0, N, tile_size):
        tile_end = min(tile_start + tile_size, N)
 
        # compute scores for this K tile 
        # S_tile = q · K_tile^T  * scale
        scores = []
        for j in range(tile_start, tile_end):
            s = sum(q[di] * K[j][di] for di in range(d)) * scale
            scores.append(s)
 
        # local softmax stats 
        m_tile = max(scores)
        p_tile = [math.exp(s - m_tile) for s in scores]   # unnormalised P
        l_tile = sum(p_tile)
 
        # online merge 
        m_new = max(m, m_tile)
        alpha  = math.exp(m - m_new)          # correction for old O
        beta   = math.exp(m_tile - m_new)     # scale for new P
 
        l_new = alpha * l + beta * l_tile
 
        # rescale O accumulator and add new contribution
        
        pv = [0.0] * d
        for idx, j in enumerate(range(tile_start, tile_end)):
            w = beta * p_tile[idx]
            for di in range(d):
                pv[di] += w * V[j][di]
 
        o = [(alpha * l * o[di] + pv[di]) / l_new for di in range(d)]
 
        m, l = m_new, l_new
 
    return o   

# 1D  Verification

def standard_attention_row(
    q: list[float],
    K: list[list[float]],
    V: list[list[float]],
    scale: float | None = None,
) -> list[float]:
    """Reference: naive O(N²) attention for one query row."""
    N = len(K)
    d = len(q)
    if scale is None:
        scale = 1.0 / math.sqrt(d)
 
    scores = [sum(q[di]*K[j][di] for di in range(d))*scale for j in range(N)]
    probs  = naive_softmax(scores)
    return [sum(probs[j]*V[j][di] for j in range(N)) for di in range(d)]
 
 
def run_tests():
    import random
    random.seed(42)
 
    print("=" * 65)
    print("SECTION 1: Online Softmax — Verification")
    print("=" * 65)
 
    # ── Test A: softmax correctness at various tile sizes ──────────────────
    print("\n[A] Online softmax vs naive softmax")
    print(f"    {'test':<25} {'tile':<6} {'max_err':<14} {'status'}")
    print(f"    {'-'*55}")
 
    test_cases = {
        "random N=8":     [random.gauss(0,1) for _ in range(8)],
        "random N=64":    [random.gauss(0,1) for _ in range(64)],
        "random N=256":   [random.gauss(0,1) for _ in range(256)],
        "all zeros N=16": [0.0]*16,
        "large values":   [random.gauss(0,100) for _ in range(32)],
        "very negative":  [random.gauss(-200,50) for _ in range(32)],
        "single elem":    [3.14],
    }
 
    all_pass = True
    for name, x in test_cases.items():
        for ts in [1, 4, 16]:
            ref = naive_softmax(x)
            got = online_softmax(x, tile_size=ts)
            err = max(abs(a-b) for a,b in zip(ref, got))
            ok  = err < 1e-12
            all_pass = all_pass and ok
            print(f"    {'PASS' if ok else 'FAIL':<4}  {name:<25} ts={ts:<3}  err={err:.2e}")


#  Test B: streaming attention row 
    print("\n[B] Streaming attention vs naive attention")
    print(f"    {'config':<30} {'max_err':<14} {'status'}")
    print(f"    {'-'*55}")
 
    configs = [
        (4,  8,  4,  "d=4  N=8   tile=4"),
        (8,  16, 4,  "d=8  N=16  tile=4"),
        (16, 64, 16, "d=16 N=64  tile=16"),
        (32, 128,32, "d=32 N=128 tile=32"),
    ]
 
    for d, N, ts, label in configs:
        q = [random.gauss(0,1) for _ in range(d)]
        K = [[random.gauss(0,1) for _ in range(d)] for _ in range(N)]
        V = [[random.gauss(0,1) for _ in range(d)] for _ in range(N)]
 
        ref = standard_attention_row(q, K, V)
        got = streaming_attention_row(q, K, V, tile_size=ts)
        err = max(abs(a-b) for a,b in zip(ref, got))
        ok  = err < 1e-10
        all_pass = all_pass and ok
        print(f"    {'PASS' if ok else 'FAIL':<4}  {label:<30} err={err:.2e}")
 
    #  Step-by-step trace 
    print("\n[C] Step-by-step online softmax trace  x=[1,2,3,4,5,6,7,8]")
    print(f"    {'tile':<6} {'m_old':>8} {'m_tile':>8} {'m_new':>8} {'l_new':>10}")
    print(f"    {'-'*50}")
    x = list(range(1, 9))
    m, l = float('-inf'), 0.0
    for step, start in enumerate(range(0, len(x), 4)):
        tile   = x[start:start+4]
        m_tile = max(tile)
        l_tile = sum(math.exp(xi - m_tile) for xi in tile)
        m_new  = max(m, m_tile)
        l_new  = math.exp(m - m_new)*l + math.exp(m_tile - m_new)*l_tile
        print(f"    T{step+1:<4}  {m:>8.2f}  {m_tile:>8.2f}  {m_new:>8.2f}  {l_new:>10.4f}")
        m, l = m_new, l_new
    ref = naive_softmax(x)
    got = online_softmax(x, tile_size=4)
    print(f"\n    naive:  {[round(v,4) for v in ref]}")
    print(f"    online: {[round(v,4) for v in got]}")
 
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("=" * 65)
 
    print("""

  1. m and l become VECTORS of shape [BLOCK_M] in the kernel
     (one entry per query in the current tile, not a single scalar).
 
  2. The rescaling  O = alpha[:,None] * O + P_tile @ V_tile
     happens IN REGISTERS — no HBM write until the loop ends.
 
  3. Storing L = m + log(l)  (logsumexp) is enough for the backward
     pass to reconstruct softmax(S) without storing N×N P.
""")
 
 
if __name__ == "__main__":
    run_tests()
 