#!/usr/bin/env python3
"""Step 3 — Evo2 genome features (the single-nucleotide gLM reference).

Evo2 models DNA at single-nucleotide resolution (StripedHyena2). For each
genome we window the cached DNA, run Evo2 with an intermediate layer returned,
pool it -> one feature vector per genome, and save an npz (bacdive_ids, features)
for model.py.

Two pooling modes (the comparison's Evo2 arms):
  --pooling mean : average Evo2's per-nucleotide embeddings (the standard
                   single-nucleotide representation)        -> evo2_features.npz
  --pooling bpe  : pool Evo2's per-nucleotide embeddings ALONG domain-BPE token
                   boundaries (each BPE "word" = mean of its bases, then mean
                   over words). The "byte-pair embeds" of Evo2 -> evo2_bpe_features.npz
                   Requires a trained domain-BPE tokenizer (--bpe-tokenizer, or
                   auto-discovered under data/tokenizers/).

REAL RUN (CUDA GPU only — Evo2 does not run on macOS/CPU):
    pip install -r requirements-evo2.txt        # on the GPU box
    python extract_evo2_features.py --model evo2_7b --layer blocks.28.mlp.l3
    python extract_evo2_features.py --pooling bpe --model evo2_7b --layer blocks.28.mlp.l3

DRY RUN on your laptop (so the rest of the pipeline is testable now):
    python extract_evo2_features.py --mock                 # k-mer stand-in
    python extract_evo2_features.py --mock --pooling bpe   # BPE-token-bag stand-in
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
from microbe_bpe.tiny_lm import bpe_pool
from microbe_bpe.tokenizers import HuggingFaceBPETokenizer

_KMER_NTS = "ACGT"


def _kmer_vector(dna: str, k: int = 4) -> np.ndarray:
    """Deterministic normalized k-mer frequency vector — the mean-pool mock.

    A legitimately weak genome representation (4^k dims), used ONLY to exercise
    the downstream pipeline without a GPU. Not Evo2; never report it as such.
    """
    idx = {nt: i for i, nt in enumerate(_KMER_NTS)}
    dim = len(_KMER_NTS) ** k
    counts = np.zeros(dim, dtype=np.float64)
    total = 0
    for i in range(len(dna) - k + 1):
        code = 0
        ok = True
        for c in dna[i : i + k]:
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


def _bpe_bag_vector(dna: str, bpe_tok: HuggingFaceBPETokenizer) -> np.ndarray:
    """Normalized histogram over BPE token ids — the --pooling bpe mock.

    A bag-of-byte-pairs stand-in (dim = BPE vocab). CPU-only, conceptually the
    'byte-pair' analogue of the k-mer mock; not Evo2.
    """
    dim = bpe_tok.vocab_size
    counts = np.zeros(dim, dtype=np.float64)
    ids = bpe_tok.encode(dna)
    for i in ids:
        if 0 <= i < dim:
            counts[i] += 1
    if counts.sum():
        counts /= counts.sum()
    return counts.astype(np.float32)


def _find_bpe_tokenizer(data_dir: Path, explicit: Path | None) -> HuggingFaceBPETokenizer:
    if explicit:
        return HuggingFaceBPETokenizer.load(explicit)
    tok_root = data_dir / "tokenizers"
    cands = sorted(
        (d for d in tok_root.glob("domain_bpe*") if (d / "tokenizer.json").exists()),
        key=lambda d: d.stat().st_mtime,
    )
    if not cands:
        sys.exit(
            f"--pooling bpe needs a trained domain-BPE tokenizer. None found under "
            f"{tok_root}. Run extract_bpe_features.py --tokenizer domain_bpe first, "
            f"or pass --bpe-tokenizer <dir>."
        )
    return HuggingFaceBPETokenizer.load(cands[-1])


def run_mock(per_genome, args, bpe_tok):
    ids, feats = [], []
    for bid, wins in per_genome:
        dna = "".join(wins)
        if args.pooling == "bpe":
            feats.append(_bpe_bag_vector(dna, bpe_tok))
        else:
            feats.append(_kmer_vector(dna, k=args.kmer_k))
        ids.append(bid)
    meta = {"mock": True, "pooling": args.pooling}
    if args.pooling == "bpe":
        meta["bpe_vocab"] = int(bpe_tok.vocab_size)
    else:
        meta["kmer_k"] = args.kmer_k
    return ids, feats, meta


def run_evo2(per_genome, args, bpe_tok):
    import torch
    from evo2 import Evo2  # noqa: F401 (import here so --mock has no CUDA dep)

    if not torch.cuda.is_available():
        sys.exit("Evo2 requires a CUDA GPU. Use --mock for a laptop dry run.")

    print(f"loading Evo2 model {args.model!r} (layer {args.layer}, pooling={args.pooling}) ...",
          flush=True)
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
            h = emb[args.layer].float().squeeze(0)   # [T, D] per-position embeddings

            if args.pooling == "bpe":
                _ids, offsets = bpe_tok.encode_with_offsets(w)
                pooled = torch.tensor(
                    bpe_pool(h.cpu().numpy(), offsets), device=h.device
                )
            else:
                pooled = h.mean(dim=0)               # [D]

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
            "pooling": args.pooling, "feature_dim": int(feat_dim) if feat_dim else None}
    if args.pooling == "bpe":
        meta["bpe_vocab"] = int(bpe_tok.vocab_size)
    return ids, feats, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--model", default="evo2_7b", help="Evo2 checkpoint (e.g. evo2_7b, evo2_1b_base)")
    p.add_argument("--layer", default="blocks.28.mlp.l3", help="intermediate layer to pool")
    p.add_argument("--pooling", choices=["mean", "bpe"], default="mean",
                   help="mean = per-nucleotide; bpe = pool along domain-BPE boundaries")
    p.add_argument("--bpe-tokenizer", type=Path, default=None,
                   help="saved domain-BPE tokenizer dir (default: newest under data/tokenizers/)")
    # windowing (defaults match Evo2's 8k base context)
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--stride", type=int, default=8192)
    p.add_argument("--max-windows", type=int, default=16, help="per genome (0 = all)")
    p.add_argument("--sampling", choices=["even", "head"], default="even",
                   help="even = windows spread across the whole genome (default)")
    p.add_argument("--mock", action="store_true", help="CPU stand-in (no GPU/Evo2)")
    p.add_argument("--kmer-k", type=int, default=4, help="mean-pool --mock: k for k-mer vector")
    args = p.parse_args()

    manifest = CorpusManifest.load(args.data_dir)
    df = manifest.ok
    if len(df) == 0:
        sys.exit("empty corpus — run build_genome_corpus.py first")

    bpe_tok = _find_bpe_tokenizer(args.data_dir, args.bpe_tokenizer) if args.pooling == "bpe" else None

    per_genome = []
    for row in df.itertuples():
        wins = window_dna(read_dna(Path(row.path)), args.window, args.stride,
                          (args.max_windows or None), sampling=args.sampling)
        if wins:
            per_genome.append((int(row.bacdive_id), wins))
    print(f"{len(per_genome)} genomes, window={args.window} stride={args.stride} "
          f"max_windows={args.max_windows} pooling={args.pooling}")

    tag = "evo2_bpe" if args.pooling == "bpe" else "evo2"
    if args.mock:
        print(f"MOCK MODE ({args.pooling}): writing a stand-in, NOT real Evo2 features.")
        ids, feats, meta = run_mock(per_genome, args, bpe_tok)
        default_name = f"{tag}_features.MOCK.npz"
    else:
        ids, feats, meta = run_evo2(per_genome, args, bpe_tok)
        default_name = f"{tag}_features.npz"

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
