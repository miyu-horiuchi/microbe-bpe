#!/usr/bin/env bash
# End-to-end smoke test on a laptop (no GPU). Uses built-in demo genomes and the
# Evo2 --mock stand-in, so it exercises the whole pipeline EXCEPT the real Evo2
# embeddings and the model.py downstream step (which needs microbe-foundation's
# BacDive data — the demo bacdive_ids won't join those labels).
#
# It proves: corpus build -> domain-BPE + single-nt feature extraction -> mock
# Evo2 features -> npz files in the right shape for run_comparison.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== 1. build demo genome corpus =="
python build_genome_corpus.py --demo --cap 60000

echo "== 2. domain-BPE features (tiny LM) =="
python extract_bpe_features.py --tokenizer domain_bpe --bpe-vocab 512 \
  --window 512 --stride 512 --max-windows 24 --steps 80 --d-model 128 --max-len 512

echo "== 3. single-nucleotide control features =="
python extract_bpe_features.py --tokenizer single_nt \
  --window 512 --stride 512 --max-windows 24 --steps 80 --d-model 128 --max-len 512

echo "== 4. Evo2 MOCK features (k-mer stand-in; NOT real Evo2) =="
python extract_evo2_features.py --mock --window 512 --stride 512 --max-windows 24

echo
echo "Smoke complete. Feature files:"
ls -la data/*features*.npz
echo
echo "Next: on a CUDA GPU box build the real corpus from microbe-foundation"
echo "accessions, run extract_evo2_features.py (no --mock), then run_comparison.py."
