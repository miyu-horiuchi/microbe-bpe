#!/usr/bin/env bash
# Real run on a rented CUDA GPU box (Lambda / RunPod / Colab A100, etc).
#
# Prereqs on the box:
#   - microbe-foundation cloned and its data built (traits/splits/vocab + accessions):
#       cd microbe-foundation
#       python fetch_bacdive.py && python parse_bacdive.py && python splits.py \
#         && python vocab.py && python extract_genome_accessions.py
#   - microbe-bpe deps:        pip install -r requirements.txt
#   - Evo2 (GPU only; Linux, Python 3.11/3.12, Torch 2.6/2.7). flash-attn must be
#     installed BEFORE evo2 and without build isolation:
#       pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
#       pip install flash-attn==2.8.0.post2 --no-build-isolation
#       pip install evo2
#       pip install -r requirements-evo2.txt   # data-side deps
#     (see https://github.com/ArcInstitute/evo2 for the authoritative install)
#
# Tune N_GENOMES / EPOCHS / SEEDS to your budget. Start small to validate.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

N_GENOMES="${N_GENOMES:-500}"
EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-0 1 2}"
EVO2_MODEL="${EVO2_MODEL:-evo2_7b}"
EVO2_LAYER="${EVO2_LAYER:-blocks.28.mlp.l3}"

# Fairness knobs for the headline pair (single_nt vs domain_bpe):
#   WINDOW <= MAXLEN  -> single-nt is never truncated, so both tokenizers see the
#                        same nucleotides over the same #steps (residue-matched).
WINDOW="${WINDOW:-512}"
MAXLEN="${MAXLEN:-512}"
MAXWIN="${MAXWIN:-128}"
STEPS="${STEPS:-1500}"

echo "== 1. shared genome corpus ($N_GENOMES genomes, FULL/uncapped) =="
python build_genome_corpus.py --limit "$N_GENOMES"

echo "== 2. single-nucleotide control features (train-split only) =="
python extract_bpe_features.py --tokenizer single_nt \
  --window "$WINDOW" --max-len "$MAXLEN" --stride $((WINDOW/2)) --max-windows "$MAXWIN" \
  --steps "$STEPS" --device cuda

echo "== 2b. k-mer control features (fixed chunks; train-split only) =="
python extract_bpe_features.py --tokenizer kmer --kmer-k 4 \
  --window "$WINDOW" --max-len "$MAXLEN" --stride $((WINDOW/2)) --max-windows "$MAXWIN" \
  --steps "$STEPS" --device cuda

echo "== 3. domain-BPE features (train-split only) =="
python extract_bpe_features.py --tokenizer domain_bpe --bpe-vocab 1024 \
  --window "$WINDOW" --max-len "$MAXLEN" --stride $((WINDOW/2)) --max-windows "$MAXWIN" \
  --steps "$STEPS" --device cuda

echo "== 4a. Evo2 features — mean pool ($EVO2_MODEL, layer $EVO2_LAYER) =="
python extract_evo2_features.py --pooling mean --model "$EVO2_MODEL" --layer "$EVO2_LAYER" \
  --window 8192 --stride 8192 --max-windows 16

echo "== 4b. Evo2 features — BPE-boundary pool (byte-pair embeds of Evo2) =="
python extract_evo2_features.py --pooling bpe --model "$EVO2_MODEL" --layer "$EVO2_LAYER" \
  --window 8192 --stride 8192 --max-windows 16

echo "== 5. intrinsic report (bits/residue, Zipf, nt/token, GC probe — GPU-free endpoint) =="
python report_intrinsic.py

echo "== 6. downstream comparison (secondary endpoint; Evo2 as reference) =="
python run_comparison.py --epochs "$EPOCHS" --splits species genus family --seeds $SEEDS

echo
echo "Done. Primary: results/intrinsic.md  |  Secondary: results/comparison.md + results/leaderboard.md"
