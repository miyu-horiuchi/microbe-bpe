"""microbe-bpe — tokenization comparison on the microbe-foundation benchmark.

Two genome representations are built over the *same* microbial genomes and fed
into microbe-foundation's feature-agnostic multi-task trait model:

    1. Evo2          — a single-nucleotide-resolution genome LM (StripedHyena2),
                       the SOTA "single-residue tokenization" reference.
    2. domain BPE    — the method from "The Tokenization Trap" paper: learn BPE
                       merges directly on microbial DNA so frequent motifs/k-mers
                       become tokens, then train a small genome LM and pool.

A matched-capacity single-nucleotide TinyGPT is also produced as the controlled
tokenization baseline (same model, only the tokenizer changes).

The comparison runs microbe-foundation's `model.py` on each feature set across
the species/genus/family splits and aggregates with `leaderboard.py`.
"""

__version__ = "0.1.0"

DNA_ALPHABET = "ACGT"
DNA_ALPHABET_N = "ACGTN"
