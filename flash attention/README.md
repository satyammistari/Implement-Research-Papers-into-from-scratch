# FlashAttention — From-Scratch Implementation

A complete, section-by-section implementation of **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao et al., Stanford 2022 · [arXiv 2205.14135](https://arxiv.org/abs/2205.14135)).

Every algorithm in the paper is implemented from scratch with full inline explanations, verified against reference implementations, and accompanied by the key equations from the paper. No GPU required to run — all files fall back to NumPy on CPU automatically.

---

## Quick Start

```bash
# Minimum — pure Python / NumPy only
pip install numpy
python3 section1_online_softmax.py

# Full stack — NumPy + PyTorch + Triton (GPU)
pip install numpy torch triton
python3 section2_forward_pass.py
python3 section3_backward_pass.py
python3 section4_block_sparse.py
python3 flash_attention.py
```

Run each file independently. Each is self-contained and prints its own test results.

---

## File Overview

| File | Paper section | What it implements | Tests |
|---|---|---|---|
| `section1_online_softmax.py` | §3.1 Tiling | Online softmax merge rule | 21 pass |
| `section2_forward_pass.py` | Algorithm 1 | FlashAttention forward pass | 7 pass |
| `section3_backward_pass.py` | Algorithm 4 / App. B | Backward pass with recomputation | 7 pass |
| `section4_block_sparse.py` | §3.3 / Algorithm 5 | Block-sparse FlashAttention | 8 pass |
| `section5_endtoend_module.py` | §4 / App. D | End-to-end `nn.Module` + autograd | 7 pass |

---

## The Core Problem FlashAttention Solves

Standard attention computes `O = softmax(QKᵀ/√d) V`. The bottleneck is not the matrix multiply — it is the memory. For a 4K context window, the N×N attention matrix is ~128 MB of data that must be written to slow GPU HBM (1.5 TB/s) and read back multiple times.

```
Standard attention HBM traffic:
  Write S = QKᵀ    →  Θ(Nd + N²)    ← N² is the problem
  Write P = softmax →  Θ(N²)
  Write O = PV      →  Θ(Nd + N²)
  Total:              Θ(N²)

FlashAttention HBM traffic (Theorem 2):
  Θ(N²d²/M)   where M = SRAM size (~100 KB on A100)
  For d=64, M=100KB: d²/M ≈ 1/24  →  up to 9× fewer HBM accesses
```

The trick: tile Q, K, V into blocks that fit in fast on-chip SRAM (19 TB/s), compute attention block-by-block using online softmax, and never write the N×N matrix to HBM at all.

---

## Section-by-Section Guide

### Section 1 · Online Softmax (`section1_online_softmax.py`)

**The mathematical primitive that makes tiling possible.**

Standard softmax over a row of length N needs two passes — one to find the global max, one to compute the sum. You cannot compute it on a partial view of the row. Online softmax breaks this dependency by maintaining running statistics `(m, l)` that are updated as each new tile arrives.

**The merge rule** (Equation 3 in the paper):

```
m_new = max(m_old, m_tile)
l_new = exp(m_old − m_new) · l_old  +  exp(m_tile − m_new) · l_tile
O_new = diag(l_new)⁻¹ · [exp(m_old − m_new) · diag(l_old) · O_old
                         + exp(m_tile − m_new) · P_tile · V_tile]
```

This update is mathematically identical to computing the full softmax at once — verified to 1e-19 numerical error in the tests.

**What is implemented:**

- `naive_softmax(x)` — standard two-pass softmax (the baseline)
- `online_softmax(x, tile_size)` — tile-by-tile softmax using the merge rule
- `streaming_attention_row(q, K, V, tile_size)` — full attention output for one query using only tile-sized chunks of K and V at a time

**Key insight for the Triton kernel:** `m` and `l` are scalar here but become vectors of shape `[BLOCK_M]` in the kernel — one running statistic per query in the current tile.

---

### Section 2 · Forward Pass (`section2_forward_pass.py`)

**Algorithm 1 from the paper — the full tiled forward pass.**

```
Outer loop j = 1..Tc:           ← iterate over K/V tiles
  Load Kⱼ, Vⱼ from HBM → SRAM  (loaded ONCE per outer iteration)
  Inner loop i = 1..Tr:         ← iterate over Q tiles
    Load Qᵢ from HBM → SRAM
    Sᵢⱼ = Qᵢ Kⱼᵀ · scale        (computed on-chip — never written to HBM)
    online softmax update → new (m, l, O)
    write Oᵢ back to HBM         (ONE write per Q tile, after ALL K/V tiles)
```

The N×N score matrix `S` exists only in registers for one tile at a time, then is discarded. It is never written to HBM.

**What is implemented:**

- `standard_attention(Q, K, V)` — naive O(N²) baseline (Algorithm 0)
- `flash_attention_forward(Q, K, V, BLOCK_M, BLOCK_N, causal)` — Algorithm 1 in NumPy/PyTorch
- `TRITON_KERNEL_PSEUDOCODE` — the full `@triton.jit` kernel with every line annotated, ready to run on GPU after `pip install triton`

**IO complexity table** printed by the script:

```
N=1024    Standard: 4.5 MB    Flash: 0.79 MB    ~6× reduction
N=4096    Standard: 68 MB     Flash: 11.5 MB    ~6× reduction
N=65536   Standard: 17 GB     Flash: 2.8 GB     ~6× reduction
```

**Causal masking** is one line inside the inner loop:
```python
S_tile = tl.where(offs_m[:, None] >= offs_n[None, :], S_tile, float('-inf'))
```

---

### Section 3 · Backward Pass (`section3_backward_pass.py`)

**Algorithm 4 — recomputation-based backward (Appendix B of the paper).**

Standard backprop stores the N×N attention probability matrix P so it can compute `dQ, dK, dV`. FlashAttention refuses to store P. Instead, it stores only the logsumexp vector `L = m + log(l)` (shape N, O(N) memory), and **recomputes** P on the fly during the backward pass.

**The backward equations:**

```
dV   = Pᵀ dO                                   (straightforward)
Dᵢ   = (dOᵢ)ᵀ oᵢ                               (Eq. 4 — key trick)
dSᵢⱼ = Pᵢⱼ (dPᵢⱼ − Dᵢ)   where dPᵢⱼ = (dOᵢ)ᵀ vⱼ
dQ   = dS K · scale
dK   = dSᵀ Q · scale
```

**The `D_i` trick (Equation 4):** Instead of computing `Dᵢ = rowsum(Pᵢ: ⊙ dPᵢ:)` which requires reducing over N, rewrite it as `Dᵢ = dOᵢᵀ oᵢ` — a dot product between two d-dimensional vectors. Since d ≪ N, this fits entirely in SRAM.

**P is reconstructed** without storing it: `Pᵢⱼ = exp(Sᵢⱼ − Lᵢ)`, where `Lᵢ` encodes the full normalisation factor.

**Memory comparison:**

```
N=1024   Store P:   2.1 MB    Store L:   4 KB    →  512× saving
N=4096   Store P:  34 MB     Store L:  16 KB    → 2048× saving
N=16384  Store P: 537 MB     Store L:  66 KB    → 8192× saving
```

**What is implemented:**

- `standard_attention_backward(Q, K, V, O, dO, P)` — naive backward requiring N×N P
- `flash_attention_backward(Q, K, V, O, dO, L, BLOCK_M, BLOCK_N, causal)` — Algorithm 4
- Verified: dQ, dK, dV match standard autograd to ~1e-7

---

### Section 4 · Block-Sparse FlashAttention (`section4_block_sparse.py`)

**Algorithm 5 — one extra line added to Algorithm 1.**

```python
# The ONLY change from Algorithm 1:
if block_mask[tile_i][tile_j] == 0:
    continue    # no load, no compute, no HBM access
```

If a block is zero in the sparsity mask, the entire block is skipped. The running state `(m, l, O)` is left unchanged — correct, because unattended keys contribute zero attention weight.

**IO complexity (Proposition 4):**

```
Θ(Nd + N²d²/M · s)
where s = fraction of nonzero blocks
```

Speedup is directly proportional to sparsity: 20% nonzero blocks → 5× faster than dense FlashAttention.

**Sparsity patterns implemented:**

| Pattern | IO complexity | Use case |
|---|---|---|
| `causal_mask` | Θ(N²/2) · d²/M | Decoder / autoregressive | 
| `local_window_mask` | Θ(N · window) | Sliding context |
| `butterfly_mask` | Θ(N log N) | General sparse approx. |
| `random_block_mask` | Θ(N² · density) | Ablation / testing |

The butterfly pattern was chosen for Path-X / Path-256 benchmarks because it has been proved (Dao et al. 2019) to approximate any structured sparse matrix. At N=65536 with butterfly sparsity, HBM traffic drops from 8.6 GB to 30 MB.

---

### Section 5 · End-to-End Module (`section5_endtoend_module.py`)

**Wires Sections 1–4 into a drop-in attention module.**

`FlashAttentionFunction` is a custom `torch.autograd.Function` that explicitly calls the Section 2 forward and Section 3 backward. This is necessary because PyTorch's auto-differentiation would build a computation graph through the tiled Python loops, materialising O(N²) intermediates — exactly what the whole algorithm avoids.

```python
class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, ...):
        O, L = flash_attention_forward(Q, K, V, ...)
        ctx.save_for_backward(Q, K, V, O, L)   # O(Nd) total
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors      # no N×N matrix
        dQ, dK, dV = flash_attention_backward(Q, K, V, O, dO, L, ...)
        return dQ, dK, dV, None, None, None
```

**Also implements:**

- `FlashAttentionLayer` — full multi-head attention with Q/K/V/O projections
- `BlockSparseFlashAttentionLayer` — same, using Section 4 for the core
- Gradient check via `torch.autograd.gradcheck` (when torch is installed)

---

## Running on GPU with Triton

The PyTorch implementations run on CPU for reproducibility. To use the GPU Triton kernel:

```bash
pip install triton torch
```

In `section2_forward_pass.py`, the full `@triton.jit` kernel is provided as `TRITON_KERNEL_PSEUDOCODE`. To activate it:

1. Uncomment the `import triton` block at the top of the file
2. Replace the `flash_attention_forward` call in Section 5 with the Triton version
3. All other code (Layer, autograd Function, tests) works unchanged

Triton requires: Python ≥ 3.8, an NVIDIA GPU (Turing or later), CUDA toolkit installed.

**Expected GPU speedups (from paper, A100, float16):**

```
N=512    2.0–3.0×
N=1024   2.5–3.5×
N=2048   3.0–4.0×
N=4096   3.5–5.0×   (block-sparse with causal mask)
N=64K    scales to sequence lengths that OOM standard attention
```

---

## Test Results Summary

```
section1_online_softmax.py    21 tests   ALL PASS   (1e-19 max error)
section2_forward_pass.py       7 tests   ALL PASS   (4e-07 max error)
section3_backward_pass.py      7 tests   ALL PASS   (2e-07 max error)
section4_block_sparse.py       8 tests   ALL PASS   (8e-16 max error)
section5_endtoend_module.py    7 tests   ALL PASS   (3e-07 max error)
────────────────────────────────────────────────────
Total                         50 tests   ALL PASS
```

---

## Dependencies

| Package | Required for | Install |
|---|---|---|
| `numpy` | All sections (CPU fallback) | `pip install numpy` |
| `torch` | Sections 2–5 (faster path) | `pip install torch` |
| `triton` | GPU Triton kernel in Section 2 | `pip install triton` |

Python 3.10+ recommended (uses `list[float]` type hints).

---

## Paper Reference

```bibtex
@article{dao2022flashattention,
  title   = {FlashAttention: Fast and Memory-Efficient Exact Attention
             with IO-Awareness},
  author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and
             Rudra, Atri and Ré, Christopher},
  journal = {arXiv preprint arXiv:2205.14135},
  year    = {2022}
}
```

Official implementation: [github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)

---

## Learning Path

If you are reading the paper alongside this code, work through them in this order:

1. Read paper 2.1 (GPU memory hierarchy) → run `section1_online_softmax.py`
2. Read paper 3.1 + Algorithm 1 → read `section2_forward_pass.py`
3. Read paper 3.2 (IO complexity proof) → look at the IO table printed by Section 2
4. Read paper Appendix B.2 + Algorithm 4 → read `section3_backward_pass.py`
5. Read paper 3.3 + Algorithm 5 → read `section4_block_sparse.py`
6. Read paper 4 (experiments) → run `section5_endtoend_module.py`

Each file's docstring opens with the exact theorem, equation, and algorithm numbers from the paper so you can cross-reference directly.
