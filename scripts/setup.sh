#!/usr/bin/env bash
# Set up microbe-bpe: populate the microbe-foundation git submodule and install
# CPU dependencies. Run Evo2 deps separately on a GPU box
# (pip install -r requirements-evo2.txt).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MF_DIR="$REPO_DIR/microbe-foundation"

# Populate the vendored submodule (no-op if already initialized). If this repo
# was cloned without --recurse-submodules, this fetches microbe-foundation.
echo "Initializing microbe-foundation submodule"
git -C "$REPO_DIR" submodule update --init --recursive

if [ ! -e "$MF_DIR/model.py" ]; then
  echo "WARNING: $MF_DIR/model.py not found — submodule may not be populated."
  echo "Try: git -C \"$REPO_DIR\" submodule update --init --recursive"
fi

echo "Installing microbe-bpe CPU dependencies"
python -m pip install -r "$REPO_DIR/requirements.txt"

echo
echo "Done. microbe-foundation submodule at: $MF_DIR"
echo "To use a different checkout instead: export MF_ROOT=/path/to/microbe-foundation"
