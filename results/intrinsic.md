# Intrinsic report — tokenization-trap diagnostics (no GPU, no labels)

Lower **bits/residue** = better compression. Higher **Zipf exp** (~1.0) = more language-like token stats. **nt/token** = mean nucleotides per token (single-nt=1; k-mer=k; domain-BPE variable). **GC R²** is a positive-control probe (cross-validated).

| Method | vocab | nt/token | Zipf exp | bits/residue | bpr held-out? | GC R² (n) |
|---|---:|---:|---:|---:|:--:|---:|
| `domain_bpe_1024` | 1024 | 4.29 | 1.117 | 1.9431 | yes | +0.999 (364) |
| `evo2_bpe` | — | — | — | — | — | +0.992 (364) |
| `evo2` | — | — | — | — | — | +0.992 (364) |
| `kmer_4` | 260 | 4.00 | 0.499 | 1.8750 | yes | +0.999 (364) |
| `single_nt` | 9 | 1.00 | 0.310 | 1.9024 | yes | +0.999 (364) |

## How to read this
- **The paper's core prediction:** `single_nt` should have ~flat token stats (low Zipf) and worse bits/residue than `domain_bpe`; `kmer` sits between. Intrinsic metrics are GPU-free and confound-free — this is the cleanest test.
- **GC R²** only sanity-checks that a representation carries decodable signal. Mock Evo2 features are k-mer/bag stand-ins, so a high GC R² there is expected and NOT evidence about real Evo2.
- Trait prediction (results/comparison.md) is the *secondary*, harder endpoint.
