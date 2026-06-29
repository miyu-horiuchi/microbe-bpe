#!/usr/bin/env bash
# Turnkey run for a fresh Lambda (or any Linux CUDA) GPU box — paste-and-go.
#
# Usage on the box:
#   git clone --recurse-submodules https://github.com/miyu-horiuchi/microbe-bpe
#   cd microbe-bpe
#   N_GENOMES=500 SEEDS="0 1 2" bash scripts/lambda_run.sh
#
# Produces:
#   results/intrinsic.md    PRIMARY  — bits/residue, Zipf, nt/token, GC probe (no GPU/labels)
#   results/comparison.md   secondary — downstream trait prediction across the ladder
#   results/leaderboard.md
#
# Tunables (env vars): N_GENOMES, SEEDS, EPOCHS, BACDIVE_END, EVO2_MODEL, EVO2_LAYER,
#                      WINDOW, MAXLEN, MAXWIN, STEPS, SKIP_EVO2 (=1 to skip GPU arms).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

N_GENOMES="${N_GENOMES:-500}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-30}"
BACDIVE_END="${BACDIVE_END:-40000}"     # fetch IDs 1..END to net N_GENOMES with accessions
EVO2_MODEL="${EVO2_MODEL:-evo2_7b}"
EVO2_LAYER="${EVO2_LAYER:-blocks.28.mlp.l3}"
WINDOW="${WINDOW:-512}"; MAXLEN="${MAXLEN:-512}"; MAXWIN="${MAXWIN:-128}"; STEPS="${STEPS:-1500}"
SKIP_EVO2="${SKIP_EVO2:-0}"

echo "== 0. TLS certs (avoids macOS/fresh-box CERTIFICATE_VERIFY_FAILED) =="
pip install -q certifi
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

echo "== 1. python deps (CPU side) =="
pip install -q -r requirements.txt

if [ "$SKIP_EVO2" != "1" ]; then
  echo "== 1b. Evo2 (GPU). flash-attn BEFORE evo2, no build isolation =="
  echo "   If this box already has evo2, skip. Adjust torch CUDA wheel to your driver."
  pip install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128 || true
  pip install -q flash-attn==2.8.0.post2 --no-build-isolation || true
  pip install -q evo2 || echo "   [warn] evo2 install failed — set SKIP_EVO2=1 or fix per ArcInstitute/evo2"
fi

echo "== 2. BacDive labels (public v2 API; resumable) =="
pushd microbe-foundation >/dev/null
python fetch_bacdive.py --start 1 --end "$BACDIVE_END" --workers 12 --sleep 0.05
python parse_bacdive.py
python splits.py
python vocab.py
python extract_genome_accessions.py
popd >/dev/null

echo "== 3. shared genome corpus ($N_GENOMES genomes, full/uncapped) =="
python build_genome_corpus.py --limit "$N_GENOMES"

DEV="cuda"; command -v nvidia-smi >/dev/null 2>&1 || DEV="cpu"
echo "== 4. matched-capacity ladder (device=$DEV; window<=max-len => residue-matched) =="
for TOK in single_nt kmer domain_bpe; do
  EXTRA=""; [ "$TOK" = "kmer" ] && EXTRA="--kmer-k 4"; [ "$TOK" = "domain_bpe" ] && EXTRA="--bpe-vocab 1024"
  python extract_bpe_features.py --tokenizer "$TOK" $EXTRA \
    --window "$WINDOW" --max-len "$MAXLEN" --stride $((WINDOW/2)) --max-windows "$MAXWIN" \
    --steps "$STEPS" --device "$DEV"
done

if [ "$SKIP_EVO2" != "1" ]; then
  echo "== 5. Evo2 reference arms (mean + BPE-boundary pool) =="
  python extract_evo2_features.py --pooling mean --model "$EVO2_MODEL" --layer "$EVO2_LAYER"
  python extract_evo2_features.py --pooling bpe  --model "$EVO2_MODEL" --layer "$EVO2_LAYER"
fi

echo "== 6. PRIMARY: intrinsic report (no GPU/labels) =="
python report_intrinsic.py

echo "== 7. secondary: downstream trait comparison =="
python run_comparison.py --epochs "$EPOCHS" --splits species genus family --seeds $SEEDS

echo
echo "Done. PRIMARY -> results/intrinsic.md  |  secondary -> results/comparison.md + leaderboard.md"
echo "Remember to TERMINATE the Lambda instance (idle time still bills)."
