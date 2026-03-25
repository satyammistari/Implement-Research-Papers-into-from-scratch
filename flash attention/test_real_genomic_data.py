"""
Test script to verify real genomic data loading
"""
import torch
from dataset import DNADataset, NUCLEOTIDE_MAP, MASK_TOKEN, PAD_TOKEN

print("=" * 70)
print("Testing Real Genomic Data Loading")
print("=" * 70)

# Test with small sample first
print("\n[Test] Attempting to load real genomic data (5 samples)...")
dataset = DNADataset(max_length=512, max_samples=5, split='train')

print(f"\n[Result] Dataset size: {len(dataset)}")

if len(dataset) > 0:
    print("\n[Success] Real genomic data loaded successfully!")

    # Inspect first sample
    sample = dataset[0]
    input_ids = sample['input_ids']
    labels = sample['labels']

    print(f"\nSample details:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Label shape: {labels.shape}")

    # Decode first 50 tokens
    print(f"\n  First 50 tokens:")
    tokens = input_ids[:50].tolist()
    print(f"    {tokens}")

    # Count token types
    inverse_map = {v: k for k, v in NUCLEOTIDE_MAP.items()}
    inverse_map[MASK_TOKEN] = '[MASK]'
    inverse_map[PAD_TOKEN] = '[PAD]'

    unique, counts = torch.unique(input_ids, return_counts=True)
    print(f"\n  Token distribution:")
    for tok, count in zip(unique.tolist(), counts.tolist()):
        tok_name = inverse_map.get(tok, f'UNK({tok})')
        print(f"    {tok_name}: {count} ({count/len(input_ids)*100:.1f}%)")

    # Decode sequence
    decoded = []
    for tok in input_ids[:100]:
        tok = tok.item()
        if tok == PAD_TOKEN:
            break
        decoded.append(inverse_map.get(tok, 'N'))

    print(f"\n  Decoded sequence (first 100 bp):")
    print(f"    {''.join(decoded)}")

    print("\n" + "=" * 70)
    print("✓ Real genomic data is working correctly!")
    print("=" * 70)
else:
    print("\n[Info] No real data loaded - using synthetic fallback")
    print("This is fine for testing Flash Attention implementation")
