#!/usr/bin/env bash
# Real run on a rented CUDA GPU box (Lambda / RunPod / Colab A100, etc).
#
# Prereqs on the box:
#   - microbe-foundation cloned and its data built (traits/splits/vocab + accessions):
#       cd microbe-foundation
#       python fetch_bacdive.py && python parse_bacdive.py && python splits.py \
#         && python vocab.py && python extract_genome_accessions.py
#   - microbe-bpe deps:        pip install -r requirements.txt
#   - Evo2 deps (GPU only):    pip install -r requirements-evo2.txt
#       (see https://github.com/ArcInstitute/evo2 for the authoritative install)
#
# Tune N_GENOMES / EPOCHS / SEEDS to your budget. Start small to validate.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

N_GENOMES="${N_GENOMES:-500}"
EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-0 1 2}"
EVO2_MODEL="${EVO2_MODEL:-evo2_7b}"
EVO2_LAYER="${EVO2_LAYER:-blocks.28.mlp.l3}"

echo "== 1. shared genome corpus ($N_GENOMES genomes) =="
python build_genome_corpus.py --limit "$N_GENOMES" --cap 200000

echo "== 2. domain-BPE features =="
python extract_bpe_features.py --tokenizer domain_bpe --bpe-vocab 1024 \
  --window 1024 --stride 512 --max-windows 64 --steps 800 --device cuda

echo "== 3. single-nucleotide control features =="
python extract_bpe_features.py --tokenizer single_nt \
  --window 1024 --stride 512 --max-windows 64 --steps 800 --device cuda

echo "== 4. Evo2 features ($EVO2_MODEL, layer $EVO2_LAYER) =="
python extract_evo2_features.py --model "$EVO2_MODEL" --layer "$EVO2_LAYER" \
  --window 8192 --stride 8192 --max-windows 16

echo "== 5. downstream comparison (Evo2 vs domain-BPE vs single-nt) =="
python run_comparison.py --epochs "$EPOCHS" --splits species genus family --seeds $SEEDS

echo
echo "Done. See results/comparison.md and results/leaderboard.md"
