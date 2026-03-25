

CONFIG = {
    # Hardware
    'device':           'cuda',
    'dtype':            'float16',
    'hbm_bandwidth_GBs': 192,       # RTX 3050 spec
    'sram_per_sm_KB':   128,         # Ampere architecture
    'num_sms':          20,

    # Model — small enough to fit 3050 at N=4096
    'embed_dim':        128,
    'num_heads':        4,
    'head_dim':         32,          # 128 / 4
    'num_layers':       3,
    'ffn_multiplier':   2,           # ffn_dim = embed_dim * 2 = 256
    'vocab_size':       7,           # A T C G N [MASK] [PAD]

    # Training
    'batch_size':       4,
    'grad_accum_steps': 4,           # effective batch = 16
    'learning_rate':    3e-4,
    'num_steps':        500,         # Reduced for testing
    'mask_prob':        0.15,        # BERT-style masking

    # Sequence lengths to test
    # Standard OOMs at N=3072 on 6GB
    # Flash handles up to N=4096 easily
    'seq_lengths_standard': [256, 512],
    'seq_lengths_flash':    [256, 512, 1024],

    # Block sizes — tuned for 3050 SRAM
    # Constraint: BLOCK_M * head_dim * 2 (fp16) << 128KB per SM
    # 32 * 32 * 2 = 2KB — very safe, gives 64 tiles per SM
    'BLOCK_M':          32,
    'BLOCK_N':          32,

    # Attention variants to compare
    'variants': ['standard', 'flash_dense', 'flash_biology',
                 'flash_dynamic', 'flash_butterfly'],

    # Dataset
    'dataset_name':     'InstaDeepAI/human_reference_genome',
    'dataset_subset':   '6144',
    'crop_length':      4096,

    # IO profiler
    'profile_every_n_steps': 50,
    'warmup_steps':           100,
}