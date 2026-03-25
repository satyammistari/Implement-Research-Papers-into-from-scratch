import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

NUCLEOTIDE_MAP = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
MASK_TOKEN     = 5
PAD_TOKEN      = 6
VOCAB_SIZE     = 7


class DNADataset(Dataset):
    def __init__(self, max_length: int = 1024,
                 mask_prob: float = 0.15,
                 split: str = 'train',
                 max_samples: int = 10_000):

        self.max_length  = max_length
        self.mask_prob   = mask_prob
        self.sequences   = []

        print(f"[Dataset] Loading DNA sequences (max_length={max_length})...")

        try:
            from datasets import load_dataset

            # Try multiple real genomic datasets in order of preference
            dataset_attempts = [
                # 1. Try genomics benchmark (Parquet format, no scripts)
                {
                    'name': 'InstaDeepAI/genomics-long-range-benchmark',
                    'config': None,
                    'field': 'sequence'
                },
                # 2. Try multi-species genomes
                {
                    'name': 'InstaDeepAI/multi_species_genomes',
                    'config': None,
                    'field': 'sequence'
                },
                # 3. Try nucleotide transformer dataset
                {
                    'name': 'InstaDeepAI/nucleotide_transformer_downstream_tasks',
                    'config': None,
                    'field': 'sequence'
                },
                # 4. Try general genomics datasets
                {
                    'name': 'songlab/genomes',
                    'config': None,
                    'field': 'sequence'
                },
            ]

            loaded = False
            for attempt in dataset_attempts:
                try:
                    print(f"[Dataset] Trying {attempt['name']}...")
                    if attempt['config']:
                        hf_data = load_dataset(
                            attempt['name'],
                            attempt['config'],
                            split=split,
                            streaming=True,
                        )
                    else:
                        hf_data = load_dataset(
                            attempt['name'],
                            split=split,
                            streaming=True,
                        )

                    for i, sample in enumerate(hf_data):
                        if i >= max_samples:
                            break
                        # Extract sequence from various possible field names
                        seq = (sample.get('sequence') or
                               sample.get('text') or
                               sample.get('seq') or
                               sample.get('dna_sequence') or
                               sample.get('genome'))
                        if seq and len(seq) > 0:
                            self.sequences.append(seq)

                    if len(self.sequences) > 0:
                        loaded = True
                        print(f"[Dataset] ✓ Successfully loaded {len(self.sequences)} sequences from {attempt['name']}")
                        break

                except Exception as dataset_error:
                    print(f"[Dataset] ✗ Failed to load {attempt['name']}: {dataset_error}")
                    continue

            if not loaded:
                raise Exception("All dataset attempts failed")

        except Exception as e:
            print(f"[Dataset] HuggingFace load failed: {e}")
            print("[Dataset] Falling back to synthetic DNA data for testing")
            self.sequences = self._synthetic_dna(max_samples, max_length)

    def _synthetic_dna(self, n: int, length: int) -> list:
        rng  = np.random.default_rng(42)
        nucs = list('ATCG')
        seqs = []
        for _ in range(n):
            seq = ''.join(rng.choice(nucs, size=length))
            seqs.append(seq)
        print(f"[Dataset] Generated {n} synthetic sequences of length {length}")
        return seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        seq  = seq[:self.max_length]
        toks = [NUCLEOTIDE_MAP.get(c, 4) for c in seq]
        toks += [PAD_TOKEN] * (self.max_length - len(toks))

        labels = toks.copy()
        masked = toks.copy()

        for i in range(len(seq)):
            if toks[i] == PAD_TOKEN:
                continue
            if np.random.random() < self.mask_prob:
                r = np.random.random()
                if r < 0.8:
                    masked[i] = MASK_TOKEN
                elif r < 0.9:
                    masked[i] = np.random.randint(0, 5)

        return {
            'input_ids': torch.tensor(masked, dtype=torch.long),
            'labels':    torch.tensor(labels, dtype=torch.long),
        }


def get_dataloader(max_length: int, batch_size: int,
                   max_samples: int = 10_000) -> DataLoader:
    dataset = DNADataset(max_length=max_length, max_samples=max_samples)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, num_workers=0, pin_memory=True)