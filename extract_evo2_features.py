#!/usr/bin/env python3
"""Step 3 — Evo2 genome features (the single-nucleotide gLM reference).

Evo2 models DNA at single-nucleotide resolution (StripedHyena2). For each
genome we window the cached DNA, run Evo2 with an intermediate layer returned,
mean-pool the layer over tokens and windows -> one feature vector per genome,
and save data/evo2_features.npz (bacdive_ids, features) for model.py.

REAL RUN (CUDA GPU only — Evo2 does not run on macOS/CPU):
    pip install -r requirements-evo2.txt        # on the GPU box
    python extract_evo2_features.py --model evo2_7b --layer blocks.28.mlp.l3 \\
        --window 8192 --max-windows 16

    evo2_7b runs in bfloat16 on any supported CUDA GPU (~15 GB).
    evo2_1b_base needs FP8 / Transformer Engine on a Hopper GPU (H100/H200).

DRY RUN on your laptop (so the rest of the pipeline is testable now):
    python extract_evo2_features.py --mock
    -> writes a clearly-labeled k-mer stand-in (NOT Evo2). Swap in the real run
       on the GPU box and re-run run_comparison.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from microbe_bpe.genome_corpus import (
    DEFAULT_DATA_DIR,
    CorpusManifest,
    read_dna,
    window_dna,
)

_KMER_NTS = "ACGT"


def _kmer_vector(dna: str, k: int = 4) -> np.ndarray:
    """Deterministic normalized k-mer frequency vector — the --mock stand-in.

    A legitimately weak genome representation (4^k dims), used ONLY to exercise
    the downstream pipeline without a GPU. It is not Evo2 and must not be
    reported as such.
    """
    idx = {nt: i for i, nt in enumerate(_KMER_NTS)}
    dim = len(_KMER_NTS) ** k
    counts = np.zeros(dim, dtype=np.float64)
    total = 0
    for i in range(len(dna) - k + 1):
        kmer = dna[i : i + k]
        code = 0
        ok = True
        for c in kmer:
            j = idx.get(c)
            if j is None:
                ok = False
                break
            code = code * 4 + j
        if ok:
            counts[code] += 1
            total += 1
    if total:
        counts /= total
    return counts.astype(np.float32)


def run_mock(per_genome, kmer_k: int):
    ids, feats = [], []
    for bid, wins in per_genome:
        dna = "".join(wins)
        feats.append(_kmer_vector(dna, k=kmer_k))
        ids.append(bid)
    dim = len(_KMER_NTS) ** kmer_k
    return ids, feats, {"mock": True, "kmer_k": kmer_k, "feature_dim": dim}


def run_evo2(per_genome, args):
    import torch
    from evo2 import Evo2  # noqa: F401 (import here so --mock has no CUDA dep)

    if not torch.cuda.is_available():
        sys.exit("Evo2 requires a CUDA GPU. Use --mock for a laptop dry run.")

    print(f"loading Evo2 model {args.model!r} (layer {args.layer}) ...", flush=True)
    model = Evo2(args.model)
    device = "cuda:0"

    ids, feats = [], []
    feat_dim = None
    start = time.time()
    for n, (bid, wins) in enumerate(per_genome, start=1):
        acc = None
        count = 0
        for w in wins:
            token_ids = torch.tensor(
                model.tokenizer.tokenize(w), dtype=torch.int
            ).unsqueeze(0).to(device)
            _, emb = model(token_ids, return_embeddings=True, layer_names=[args.layer])
            h = emb[args.layer].float()           # [1, T, D]
            pooled = h.mean(dim=1).squeeze(0)      # [D]
            acc = pooled if acc is None else acc + pooled
            count += 1
        if count == 0:
            continue
        vec = (acc / count).detach().cpu().numpy().astype(np.float32)
        feat_dim = vec.shape[0]
        ids.append(bid)
        feats.append(vec)
        if n % 25 == 0:
            print(f"  [{n}/{len(per_genome)}] {n/max(time.time()-start,1e-6):.2f}/s", flush=True)
    meta = {"mock": False, "model": args.model, "layer": args.layer,
            "feature_dim": int(feat_dim) if feat_dim else None}
    return ids, feats, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out", type=Path, default=None, help="default data/evo2_features.npz")
    p.add_argument("--model", default="evo2_7b", help="Evo2 checkpoint (e.g. evo2_7b, evo2_1b_base)")
    p.add_argument("--layer", default="blocks.28.mlp.l3", help="intermediate layer to pool")
    # windowing (defaults chosen to match Evo2's 8k base context)
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--stride", type=int, default=8192)
    p.add_argument("--max-windows", type=int, default=16, help="per genome (0 = all)")
    p.add_argument("--mock", action="store_true", help="CPU k-mer stand-in (no GPU/Evo2)")
    p.add_argument("--kmer-k", type=int, default=4, help="--mock only: k for k-mer vector")
    args = p.parse_args()

    manifest = CorpusManifest.load(args.data_dir)
    df = manifest.ok
    if len(df) == 0:
        sys.exit("empty corpus — run build_genome_corpus.py first")

    per_genome = []
    for row in df.itertuples():
        wins = window_dna(read_dna(Path(row.path)), args.window, args.stride,
                          (args.max_windows or None))
        if wins:
            per_genome.append((int(row.bacdive_id), wins))
    print(f"{len(per_genome)} genomes, window={args.window} stride={args.stride} "
          f"max_windows={args.max_windows}")

    if args.mock:
        print("MOCK MODE: writing a k-mer stand-in, NOT real Evo2 features.")
        ids, feats, meta = run_mock(per_genome, args.kmer_k)
        default_name = "evo2_features.MOCK.npz"
    else:
        ids, feats, meta = run_evo2(per_genome, args)
        default_name = "evo2_features.npz"

    out = args.out or (args.data_dir / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, bacdive_ids=np.array(ids, dtype=np.int64),
             features=np.stack(feats).astype(np.float32))
    meta.update({"n_genomes": len(ids), "window": args.window,
                 "stride": args.stride, "max_windows": args.max_windows})
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    dim = feats[0].shape[0] if feats else 0
    print(f"\nwrote {out}  [{len(ids)}, {dim}]")
    if args.mock:
        print("NOTE: re-run without --mock on a CUDA GPU box for real Evo2 features.")


if __name__ == "__main__":
    main()
