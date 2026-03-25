

import os
import math
import time
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

from config  import CONFIG
from profiler import RTX3050Profiler
from dataset  import get_dataloader
from model    import TinyDNATransformer
from forward_pass import flash_attention_forward
from block_sparse import (
    block_sparse_flash_attention,
    biology_informed_mask, butterfly_mask
)

DEVICE = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
print(f"[Experiment] Running on: {DEVICE}")
if torch.cuda.is_available():
    print(f"[Experiment] GPU: {torch.cuda.get_device_name()}")
    print(f"[Experiment] VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")



# Phase 1: Memory wall demo


def run_memory_wall_demo():
    """
    Shows exactly where standard attention OOMs on 6GB.
    Produces Figure 1 of your experiment.
    """
    print("\n" + "="*60)
    print("PHASE 1: Memory Wall Demo")
    print("="*60)

    results = []
    B, H, d = 2, 4, 32

    for N in CONFIG['seq_lengths_flash']:
        for variant in ['standard', 'flash']:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                Q = torch.randn(B,H,N,d, device=DEVICE, dtype=torch.float16)
                K = torch.randn(B,H,N,d, device=DEVICE, dtype=torch.float16)
                V = torch.randn(B,H,N,d, device=DEVICE, dtype=torch.float16)

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()

                if variant == 'standard':
                    sc = 1.0 / math.sqrt(d)
                    S  = (Q.float() @ K.float().transpose(-2,-1)) * sc
                    P  = torch.softmax(S, dim=-1)
                    O  = P @ V.float()
                else:
                    O = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000
                if torch.cuda.is_available():
                    mem_mb  = torch.cuda.max_memory_allocated() / 1e6
                else:
                    mem_mb = 0
                status  = 'OK'

            except torch.cuda.OutOfMemoryError:
                elapsed = float('inf')
                mem_mb  = CONFIG['device'] and 6144 or 0
                status  = 'OOM'

            finally:
                try: del Q, K, V, S, P, O
                except: pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results.append({
                'variant': variant, 'N': N,
                'time_ms': elapsed, 'mem_mb': mem_mb, 'status': status
            })
            print(f"  {variant:10s} N={N:5d}  "
                  f"{'OOM' if status=='OOM' else f'{elapsed:7.2f}ms'}  "
                  f"{mem_mb:8.1f}MB  {status}")

    # Save plot
    _plot_memory_wall(results)
    return results


def _plot_memory_wall(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for variant, color in [('standard', 'coral'), ('flash', 'steelblue')]:
        data = [r for r in results if r['variant']==variant and r['status']=='OK']
        Ns   = [r['N']      for r in data]
        mems = [r['mem_mb'] for r in data]
        tms  = [r['time_ms'] for r in data]

        ax1.plot(Ns, mems, 'o-', color=color, label=variant, linewidth=2)
        ax2.plot(Ns, tms,  'o-', color=color, label=variant, linewidth=2)

    ax1.axhline(y=6144, color='red', linestyle='--', label='6GB limit')
    ax1.set_xlabel('Sequence length N'); ax1.set_ylabel('Peak VRAM (MB)')
    ax1.set_title('Memory usage: standard vs FlashAttention')
    ax1.legend(); ax1.set_yscale('log')

    ax2.set_xlabel('Sequence length N'); ax2.set_ylabel('Time per forward (ms)')
    ax2.set_title('Speed: standard vs FlashAttention')
    ax2.legend(); ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('results/phase1_memory_wall.png', dpi=150)
    print("\n[Phase 1] Plot saved: results/phase1_memory_wall.png")
    plt.close()

# Phase 2: IO profiling


def run_io_profiling():
    print("\n" + "="*60)
    print("PHASE 2: IO Profiler — Theory vs Empirical")
    print("="*60)

    all_summaries = []
    d = CONFIG['head_dim']

    for N in [512, 1024, 2048, 4096]:
        for variant_name in ['flash_dense', 'flash_biology', 'flash_butterfly']:
            profiler = RTX3050Profiler(d=d, variant=variant_name, N=N)

            B, H = 2, CONFIG['num_heads']
            Q = torch.randn(B,H,N,d, device=DEVICE, dtype=torch.float16)
            K = torch.randn(B,H,N,d, device=DEVICE, dtype=torch.float16)
            V = torch.randn(B,H,N,d, device=DEVICE, dtype=torch.float16)

            Q_np = Q.cpu().float().numpy()
            K_np = K.cpu().float().numpy()
            V_np = V.cpu().float().numpy()

            for _ in range(20):   # 20 measurements per config
                if variant_name == 'flash_dense':
                    profiler.record(
                        flash_attention_forward, Q_np, K_np, V_np,
                        CONFIG['BLOCK_M'], CONFIG['BLOCK_N']
                    )
                elif variant_name == 'flash_biology':
                    mask = biology_informed_mask(N, CONFIG['BLOCK_M'], CONFIG['BLOCK_N'])
                    profiler.record(
                        block_sparse_flash_attention, Q_np, K_np, V_np,
                        mask, CONFIG['BLOCK_M'], CONFIG['BLOCK_N']
                    )
                elif variant_name == 'flash_butterfly':
                    mask = butterfly_mask(N, CONFIG['BLOCK_M'], CONFIG['BLOCK_N'])
                    profiler.record(
                        block_sparse_flash_attention, Q_np, K_np, V_np,
                        mask, CONFIG['BLOCK_M'], CONFIG['BLOCK_N']
                    )

            profiler.print_report()
            all_summaries.append(profiler.summary())

            del Q, K, V, Q_np, K_np, V_np
            torch.cuda.empty_cache()

    with open('results/phase2_io_profile.json', 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print("\n[Phase 2] Results saved: results/phase2_io_profile.json")
    return all_summaries


# Phase 3: DNA training


def train_one_variant(attn_type: str, seq_length: int,
                      num_steps: int) -> list:
    """Train TinyDNATransformer with one attention variant."""

    print(f"\n[Train] variant={attn_type}  N={seq_length}  steps={num_steps}")

    model = TinyDNATransformer(
        attn_type  = attn_type,
        embed_dim  = CONFIG['embed_dim'],
        num_heads  = CONFIG['num_heads'],
        num_layers = CONFIG['num_layers'],
        max_len    = seq_length,
        BLOCK_M    = CONFIG['BLOCK_M'],
        BLOCK_N    = CONFIG['BLOCK_N'],
    ).to(DEVICE)

    print(f"[Train] Model parameters: {model.num_parameters:,}")

    optim    = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                                  weight_decay=0.01)
    loss_fn  = nn.CrossEntropyLoss(ignore_index=6)   # ignore PAD
    loader   = get_dataloader(seq_length, CONFIG['batch_size'])

    results  = []
    step     = 0

    for batch in loader:
        if step >= num_steps:
            break

        input_ids = batch['input_ids'].to(DEVICE)
        labels    = batch['labels'].to(DEVICE)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        logits = model(input_ids)
        loss   = loss_fn(logits.reshape(-1, 7), labels.reshape(-1))
        loss.backward()

        if (step + 1) % CONFIG['grad_accum_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        if torch.cuda.is_available():
            mem_mb  = torch.cuda.max_memory_allocated() / 1e6
        else:
            mem_mb = 0

        if step % 100 == 0:
            ppl = math.exp(min(loss.item(), 10))
            results.append({
                'step':     step,
                'ppl':      round(ppl, 4),
                'loss':     round(loss.item(), 4),
                'mem_mb':   round(mem_mb, 1),
                'time_ms':  round(elapsed, 2),
            })
            print(f"  step={step:5d}  ppl={ppl:6.3f}  "
                  f"mem={mem_mb:7.1f}MB  time={elapsed:6.1f}ms")

        step += 1

    return results


def run_dna_training():
    print("\n" + "="*60)
    print("PHASE 3: DNA Training — All Variants")
    print("="*60)

    N = 1024   # safe for all variants including standard
    steps = CONFIG['num_steps']
    variants = ['standard', 'flash_dense', 'flash_biology',
                'flash_butterfly', 'flash_dynamic']

    all_results = {}
    for variant in variants:
        try:
            res = train_one_variant(variant, N, steps)
            all_results[variant] = res
        except torch.cuda.OutOfMemoryError:
            print(f"[Train] {variant} OOM at N={N}")
            all_results[variant] = []

    # Save results
    with open('results/phase3_training.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    _plot_training_curves(all_results)
    return all_results


def _plot_training_curves(all_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['gray', 'steelblue', 'green', 'purple', 'orange']

    for (variant, records), color in zip(all_results.items(), colors):
        if not records: continue
        steps = [r['step']   for r in records]
        ppls  = [r['ppl']    for r in records]
        mems  = [r['mem_mb'] for r in records]

        ax1.plot(steps, ppls,  label=variant, color=color, linewidth=2)
        ax2.plot(steps, mems,  label=variant, color=color, linewidth=2)

    ax1.set_xlabel('Training step'); ax1.set_ylabel('Perplexity')
    ax1.set_title('DNA Masked Prediction: Learning Curves')
    ax1.legend(); ax1.set_ylim(bottom=1)

    ax2.set_xlabel('Training step'); ax2.set_ylabel('Peak VRAM (MB)')
    ax2.set_title('Memory Usage During Training')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/phase3_training_curves.png', dpi=150)
    print("\n[Phase 3] Plot saved: results/phase3_training_curves.png")
    plt.close()


# Main


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*60)
    print("FlashAttention × DNA Genomics Experiment")
    print("RTX 3050 6GB Edition")
    print("="*60)

    # Phase 1: ~30 minutes
    mem_results = run_memory_wall_demo()

    # Phase 2: ~1 hour
    io_results = run_io_profiling()

    # Phase 3: ~3-4 hours
    train_results = run_dna_training()

    print("\n" + "="*60)
    print("ALL PHASES COMPLETE")
    print("Results saved in: ./results/")
    print("  phase1_memory_wall.png")
    print("  phase2_io_profile.json")
    print("  phase3_training_curves.png")
    print("  phase3_training.json")
    print("="*60)


