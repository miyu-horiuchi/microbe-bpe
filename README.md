# microbe-bpe

**Does the tokenization trap matter downstream? Evo2 (single-nucleotide gLM) vs domain-adaptive BPE on the microbe-foundation trait benchmark.**

This repo connects two prior projects:

- [**microbe-foundation**](https://github.com/miyu-horiuchi/microbe-foundation) — a feature-agnostic, multi-task benchmark that predicts 21 microbial traits from genome features, with strict species/genus/family holdouts and a leaderboard.
- [**BPE / "The Tokenization Trap"**](https://github.com/miyu-horiuchi/BPE) — the claim that single-residue tokenization gives protein/genome LMs a flat, non-Zipfian token distribution that starves scaling, and that *domain-adaptive BPE* restores language-like statistics and better modelling at fixed size.

The paper's evidence is intrinsic (Zipf exponents, bits-per-residue, a synthetic probe). **microbe-bpe asks the downstream question:** on real microbial genomes, does the tokenizer change what a model can predict about an organism? We compare two genome representations on the same genomes and the same trait heads:

| Method | Tokenization | Scale | Role |
|---|---|---|---|
| **Evo2** | single nucleotide (StripedHyena2) | 1B–40B, pretrained on 8.8T tokens | SOTA single-residue gLM reference |
| **domain BPE** | BPE merges learned on microbial DNA | tiny TinyGPT, trained here | the tokenization-trap method |
| single-nt (control) | single nucleotide | same TinyGPT as BPE | matched-capacity baseline |

The single-nt TinyGPT control isolates *tokenization at matched capacity* (only the tokenizer changes); Evo2 is the heavyweight single-nucleotide reference. The headline comparison the task asks for is **Evo2 vs domain-BPE**.

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
                              └─▶ data/domain_bpe_*_features.npz          └─▶ data/evo2_features.npz
                                  data/single_nt_features.npz
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

Builds a tiny demo corpus (6 reference genomes), trains the domain-BPE and single-nt TinyGPTs, and writes a **mock** Evo2 feature file — proving the pipeline end-to-end. The demo `bacdive_id`s are synthetic, so they don't join microbe-foundation's labels; the downstream `model.py` step needs the real corpus below.

### 2. Real comparison (on a CUDA GPU box)

First build microbe-foundation's benchmark data (once) inside the submodule:

```bash
cd microbe-foundation
python fetch_bacdive.py && python parse_bacdive.py && python splits.py \
  && python vocab.py && python extract_genome_accessions.py
cd ..
pip install -r requirements.txt          # CPU deps
pip install -r requirements-evo2.txt     # Evo2 (GPU only; see ArcInstitute/evo2)
```

Then run the whole comparison (tune size to your budget):

```bash
N_GENOMES=500 EPOCHS=30 SEEDS="0 1 2" bash scripts/run_evo2_gpu.sh
```

or step by step:

```bash
python build_genome_corpus.py --limit 500 --cap 200000
python extract_bpe_features.py --tokenizer domain_bpe --bpe-vocab 1024 --device cuda
python extract_bpe_features.py --tokenizer single_nt --device cuda
python extract_evo2_features.py --model evo2_7b --layer blocks.28.mlp.l3
python run_comparison.py --epochs 30 --splits species genus family --seeds 0 1 2
```

Outputs land in `results/comparison.md` (Evo2-vs-BPE head-to-head) and `results/leaderboard.md` (microbe-foundation's leaderboard over all runs).

---

## What the comparison reports

- **Mean per-head test score** for each method on each split (species/genus/family), seed-averaged.
- **Per-head Δ (domain-BPE − Evo2)** so you can see *which traits* the tokenizer choice helps — the paper predicts the gap concentrates on motif/k-mer-decided ("machinery") phenotypes and shrinks under covariate shift.
- microbe-foundation's standard **leaderboard** ranking every run.

A bonus bits-per-residue diagnostic is logged in each BPE/single-nt `*.meta.json` (the paper's intrinsic metric), though the headline result here is downstream trait F1.

---

## Honest framing

- **Capacity is not matched between Evo2 and BPE.** Evo2 is a multi-billion-parameter model pretrained on 8.8T tokens; the domain-BPE model is a tiny from-scratch TinyGPT. The fair tokenization-only test is **single-nt TinyGPT vs domain-BPE TinyGPT** (matched capacity); Evo2 is the SOTA single-nt *reference ceiling*. Read all three together.
- **Frozen features, linear-ish heads.** Both representations are frozen and fed to microbe-foundation's MLP heads, so we measure representational content, not end-to-end finetuning.
- **Unsupervised pretraining.** The TinyGPTs are pretrained on genome DNA (no trait labels), mirroring Evo2's external pretraining. Use `--train-split train` for a stricter no-leakage protocol that pretrains only on the training genomes of a split.
- **Genomes are length-capped** (default 200 kb) for tractable dev runs; pass `--cap 0` for full genomes.

## Repository layout

```
microbe_bpe/
  genome_corpus.py     # fetch + window + cache genome DNA (shared input)
  tokenizers.py        # single-nt + domain-adaptive BPE (DNA), from the BPE repo
  tiny_lm.py           # TinyGPT genome LM + pooled feature extraction
  mf_bridge.py         # locate/reuse the microbe-foundation checkout
build_genome_corpus.py # step 1: shared DNA corpus
extract_bpe_features.py# step 2: domain-BPE + single-nt features
extract_evo2_features.py# step 3: Evo2 features (GPU) or --mock (CPU)
run_comparison.py      # step 4: drive model.py + leaderboard, summarize
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
