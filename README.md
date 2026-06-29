# microbe-bpe

**Does the tokenization trap matter downstream? Evo2 (single-nucleotide gLM) vs domain-adaptive BPE on the microbe-foundation trait benchmark.**

This repo connects two prior projects:

- [**microbe-foundation**](https://github.com/miyu-horiuchi/microbe-foundation) — a feature-agnostic, multi-task benchmark that predicts 21 microbial traits from genome features, with strict species/genus/family holdouts and a leaderboard.
- [**BPE / "The Tokenization Trap"**](https://github.com/miyu-horiuchi/BPE) — the claim that single-residue tokenization gives protein/genome LMs a flat, non-Zipfian token distribution that starves scaling, and that *domain-adaptive BPE* restores language-like statistics and better modelling at fixed size.

The paper's claim is fundamentally about **language-model efficiency**, so microbe-bpe reports two tiers of evidence on the same models:

- **Primary (intrinsic, GPU-free, label-free):** does domain-BPE compress microbial DNA better (lower **bits-per-residue**) with more language-like (**Zipfian**) token statistics than single-nt, with k-mers in between? Plus a **GC-content positive-control probe**. See `report_intrinsic.py → results/intrinsic.md`. This is the cleanest, confound-free test and runs on a laptop.
- **Secondary (downstream):** does any intrinsic advantage *transfer* to predicting microbial traits? This is harder and noisier (organism traits are largely about gene content), so it's a transfer check, not the headline.

To keep both *causal*, the comparison is a **matched-capacity A/B test where only the tokenizer differs**:

| Method | Tokenization | Scale | Role |
|---|---|---|---|
| **single-nt** | single nucleotide | tiny TinyGPT, trained here | **headline control** (the "trap") |
| **k-mer** | fixed non-overlapping k-base chunks | **same TinyGPT, same training** | **fair middle rung** (chunks, but not *learned*) |
| **domain BPE** | BPE merges learned on microbial DNA | **same TinyGPT, same training** | **headline method** (the tokenization-trap claim) |
| Evo2 (mean) | single nucleotide (StripedHyena2) | 1B–40B, pretrained on 8.8T tokens | reference ceiling (single-residue) |
| Evo2-BPE | Evo2 embeddings pooled along BPE word boundaries | same Evo2 | reference: does BPE composition help a single-nt model? |

The `single-nt → k-mer → domain-BPE` ladder is deliberate: it separates *"chunks beat single letters"* from *"**learned** chunks beat fixed chunks"* — the latter is the paper's actual claim.

Because `single-nt` and `domain BPE` share the same architecture, the same windows, and **residue-matched** training (see below), the *only* difference between them is how the DNA is chopped — so any downstream gap is attributable to tokenization, not scale/data/architecture. Evo2 (billions of params, 8.8T-token pretraining) is reported as a *reference*, never as a controlled comparison.

**Evo2-BPE** is the same Evo2 forward pass, but instead of averaging its per-nucleotide embeddings, we mean-pool them *within each domain-BPE token span* and then over tokens — so Evo2's representation is composed at "word" granularity. It isolates whether BPE-style composition helps even a model trained per-nucleotide (mirrors the paper's "pool along BPE merge boundaries" proposal).

> ⚠️ **Evo2 needs a CUDA GPU** (it will not run on macOS/CPU). `evo2_7b` runs in bfloat16 on any supported CUDA GPU (~15 GB); `evo2_1b_base` needs FP8/Transformer Engine on a Hopper GPU. Everything else (corpus build, BPE/single-nt feature extraction, the comparison driver) runs on a laptop. A CPU `--mock` stand-in lets you dry-run the full pipeline before renting a GPU.

---

## How it fits together

```
microbe-foundation (git submodule)             microbe-bpe (this repo)
  data/genome_accessions.tsv  ────────────────▶  build_genome_corpus.py
  data/splits.parquet                              └─▶ data/genome_dna/<bid>.txt.gz  (one DNA string per genome)
                                                       data/genome_manifest.parquet
                                                            │  (identical DNA for both methods)
                                       ┌────────────────────┴─────────────────────┐
                            extract_bpe_features.py                    extract_evo2_features.py
                            (domain BPE + single-nt TinyGPT)           (Evo2 on GPU, or --mock)
                              └─▶ data/domain_bpe_*_features.npz          ├─▶ data/evo2_features.npz      (--pooling mean)
                                  data/single_nt_features.npz             └─▶ data/evo2_bpe_features.npz  (--pooling bpe)
                                                            │
                                                  run_comparison.py
                                       (runs microbe-foundation model.py per
                                        method × split × seed, then leaderboard.py)
                                                            │
                                          results/comparison.md  +  results/leaderboard.md
```

Every feature file is just `bacdive_ids [N]` + `features [N, D]` — exactly what microbe-foundation's `model.py --features` consumes, so the comparison is apples-to-apples on the downstream heads, with the genome representation as the only thing that varies.

---

## Quick start

### 0. Setup (populates the microbe-foundation submodule, installs CPU deps)

```bash
# if you didn't clone with --recurse-submodules:
git submodule update --init --recursive
bash scripts/setup.sh
# want to point at a different checkout instead? export MF_ROOT=/path/to/microbe-foundation
```

microbe-foundation is vendored as a git submodule at `microbe-foundation/`. Clone
this repo with `git clone --recurse-submodules <url>` to get it in one step.

### 1. Laptop smoke test (no GPU, no BacDive data)

```bash
bash scripts/run_smoke.sh
```

Builds a tiny demo corpus (6 reference genomes), trains the single-nt / k-mer / domain-BPE TinyGPTs, writes **mock** Evo2 feature files, and prints `results/intrinsic.md` — proving the pipeline end-to-end (even on the demo, the Zipf exponent already shows single-nt ≈ flat vs domain-BPE ≈ language-like). The demo `bacdive_id`s are synthetic, so they don't join microbe-foundation's labels; the downstream `model.py` step needs the real corpus below.

### 2. Real comparison (on a CUDA GPU box)

First build microbe-foundation's benchmark data (once) inside the submodule:

```bash
cd microbe-foundation
python fetch_bacdive.py && python parse_bacdive.py && python splits.py \
  && python vocab.py && python extract_genome_accessions.py
cd ..
pip install -r requirements.txt          # CPU deps
# Evo2 (GPU only; Linux + Python 3.11/3.12 + Torch 2.6/2.7). flash-attn must be
# installed before evo2 and without build isolation — it is NOT a plain -r install:
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
pip install -r requirements-evo2.txt     # data-side deps (see ArcInstitute/evo2)
```

Then run the whole comparison (tune size to your budget):

```bash
N_GENOMES=500 EPOCHS=30 SEEDS="0 1 2" bash scripts/run_evo2_gpu.sh
```

or step by step:

```bash
python build_genome_corpus.py --limit 500          # full (uncapped) genomes
# matched-capacity ladder — --window <= --max-len keeps them residue-matched, and
# --train-split train (the default) means no test genomes leak into pretraining:
python extract_bpe_features.py --tokenizer single_nt  --window 512 --max-len 512 --steps 1500 --device cuda
python extract_bpe_features.py --tokenizer kmer --kmer-k 4 --window 512 --max-len 512 --steps 1500 --device cuda
python extract_bpe_features.py --tokenizer domain_bpe --bpe-vocab 1024 --window 512 --max-len 512 --steps 1500 --device cuda
# references (Evo2):
python extract_evo2_features.py --pooling mean --model evo2_7b --layer blocks.28.mlp.l3
python extract_evo2_features.py --pooling bpe  --model evo2_7b --layer blocks.28.mlp.l3
python report_intrinsic.py                          # PRIMARY: results/intrinsic.md
python run_comparison.py --epochs 30 --splits species genus family --seeds 0 1 2  # secondary
```

Outputs: **`results/intrinsic.md`** (primary — bits/residue, Zipf, nt/token, GC probe per method), `results/comparison.md` (secondary — trait prediction across the ladder + Evo2 references), and `results/leaderboard.md`.

---

## What gets reported

**Primary — `results/intrinsic.md` (`report_intrinsic.py`, no GPU/labels):**
- **bits-per-residue** (held-out when a split is set), **Zipf exponent**, and **nt/token** (compression) per method — the paper's actual claim, tested confound-free.
- **GC-content probe**: cross-validated R² predicting each genome's GC from its frozen features — a positive control that signal exists.

**Secondary — `results/comparison.md` (`run_comparison.py`):**
- **Mean per-head test score** for each method on each split (species/genus/family), reported as **mean ± std over seeds** so you can see the noise floor.
- **Headline contrast Δ(domain_bpe − single_nt) by trait class**: the per-head gap, aggregated over *machinery* traits (gene-content / functional: pathogenicity, AMR, cultivation, metabolites…) vs *compositional* traits (bulk-sequence: gram stain, oxygen, temperature…). The paper predicts any tokenization benefit concentrates on machinery phenotypes — averaging all 21 heads together would wash that out, so we break it out. A two-sided **Wilcoxon signed-rank p-value** across the heads in each class is reported when `scipy` is installed and there are ≥6 heads.
- **Full per-head table** for the headline pair (plus the Evo2 references) on the species split.
- microbe-foundation's standard **leaderboard** ranking every run.

A bonus bits-per-residue diagnostic (the paper's intrinsic compression metric) is logged in each BPE/single-nt `*.meta.json`, alongside `residue_matched` and `approx_residues_seen` for auditing fairness — though the headline result here is downstream trait prediction.

---

## Honest framing

- **The headline test is matched-capacity and residue-matched.** `single_nt` and `domain_bpe` are the *same* TinyGPT trained on the *same* windows for the *same* number of steps. Keeping `--window <= --max-len` means the single-nt model is never truncated, so both tokenizers see exactly the same nucleotides — they just chop them differently. The only confound left is the tokenizer, which is the point.
- **Evo2 is a reference, not a control.** It is a multi-billion-parameter model pretrained on 8.8T tokens, so any Evo2-vs-TinyGPT gap mixes scale + data + architecture with tokenization. We report it as a ceiling, never as a clean comparison.
- **No test peeking.** `--train-split train` is the **default**: both the BPE merges and the LM are fit only on the training genomes of `--split-level`. Features are still extracted for all genomes (extraction is unsupervised — no labels involved).
- **Representative windows, full genomes.** Genomes are cached uncapped by default and windows are sampled *evenly across the whole genome* (`--sampling even`), not just the leading contigs. Use `--cap`/`--sampling head` only for cheap dev runs.
- **Frozen features.** Both representations are frozen and fed to microbe-foundation's MLP heads, so we measure representational content, not end-to-end finetuning.
- **Expect a weak signal.** A tiny LM over raw DNA windows is a screening-grade representation; treat absolute numbers as directional and lean on the per-trait-class Δ and seed error bars, plus the intrinsic bits-per-residue diagnostic.

## Repository layout

```
microbe_bpe/
  genome_corpus.py     # fetch + window + cache genome DNA (shared input)
  tokenizers.py        # single-nt + k-mer + domain-adaptive BPE (DNA)
  tiny_lm.py           # TinyGPT genome LM + pooled feature extraction
  intrinsic.py         # bits/residue helpers, Zipf exponent, compression, GC
  mf_bridge.py         # locate/reuse the microbe-foundation checkout
build_genome_corpus.py # step 1: shared DNA corpus
extract_bpe_features.py# step 2: single-nt / k-mer / domain-BPE features (+intrinsic meta)
extract_evo2_features.py# step 3: Evo2 features (GPU) or --mock (CPU)
report_intrinsic.py    # PRIMARY endpoint: intrinsic table + GC probe (no GPU/labels)
run_comparison.py      # secondary endpoint: drive model.py + leaderboard, summarize
microbe-foundation/    # git submodule: the benchmark, model.py, leaderboard.py
scripts/               # setup.sh, run_smoke.sh, run_evo2_gpu.sh
tests/                 # CPU-only unit tests
```

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Credit / license

Builds directly on `microbe-foundation` and the `BPE` ("Tokenization Trap") repo by M. Horiuchi. Evo2 by the Arc Institute (StripedHyena2). Code MIT; BacDive/MediaDive data are CC-BY 4.0.
