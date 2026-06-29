#!/usr/bin/env python3
"""Step 2 — domain-BPE (and single-nucleotide control) genome features.

Implements the paper's method on real microbial genomes:
  1. learn a domain-adaptive BPE tokenizer on the cached genome DNA windows,
  2. train a small TinyGPT genome LM under that tokenizer,
  3. freeze it and mean-pool hidden states over each genome's windows -> one
     feature vector per genome,
  4. save data/<tokenizer>_features.npz (bacdive_ids, features) for model.py.

Run it once per tokenizer (the architecture is identical; only the tokenizer
changes), so the single-nt vs domain-BPE comparison is matched-capacity. This is
the HEADLINE comparison of the experiment (Evo2 is only a far-larger reference):

    python extract_bpe_features.py --tokenizer single_nt
    python extract_bpe_features.py --tokenizer domain_bpe --bpe-vocab 1024

Everything here runs on CPU (laptop OK) for a dev-sized corpus; a GPU just makes
the LM training faster (--device cuda).

Fairness protocol (addresses the obvious confounds):
  * Residue-matched, not just step-matched. Both tokenizers train on the SAME
    windows for the SAME number of steps; with --window <= --max-len the single-nt
    model is never truncated, so both models see exactly the same nucleotides.
    The only difference is the chopping (the whole point of the paper).
  * No test peeking. --train-split defaults to `train`: the tokenizer AND the LM
    are fit only on the training genomes of --split-level. Features are still
    extracted for every genome (extraction is unsupervised, no labels involved).
  * Representative windows. We sample windows spread across the WHOLE genome
    (--sampling even), not just the leading contigs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from microbe_bpe.genome_corpus import (
    DEFAULT_DATA_DIR,
    CorpusManifest,
    read_dna,
    window_dna,
)
from microbe_bpe.tiny_lm import LMConfig, eval_bits_per_residue, genome_embedding, train_lm
from microbe_bpe.tokenizers import DomainBPETrainer, NucleotideTokenizer


def gather_windows(
    manifest: CorpusManifest,
    *,
    window: int,
    stride: int,
    max_windows_per_genome: int,
    split: str | None,
    split_level: str | None,
    sampling: str = "even",
) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """Return (training_windows, [(bacdive_id, genome_windows), ...]).

    `training_windows` is the flat pool used to train the tokenizer + LM
    (optionally restricted to one split). The per-genome list is used for
    feature extraction and always covers every cached genome.
    """
    df = manifest.ok
    split_col = f"{split_level}_split" if split_level else None

    train_pool: list[str] = []
    per_genome: list[tuple[int, list[str]]] = []
    for row in df.itertuples():
        bid = int(row.bacdive_id)
        dna = read_dna(Path(row.path))
        wins = window_dna(dna, window, stride, max_windows_per_genome, sampling=sampling)
        if not wins:
            continue
        per_genome.append((bid, wins))
        in_train = True
        if split and split_col and hasattr(row, split_col):
            in_train = getattr(row, split_col) == split
        if in_train:
            train_pool.extend(wins)
    return train_pool, per_genome


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", choices=["domain_bpe", "single_nt"], default="domain_bpe")
    p.add_argument("--bpe-vocab", type=int, default=1024, help="domain-BPE vocab size")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out", type=Path, default=None, help="output npz (default data/<tok>_features.npz)")
    # windowing — keep --window <= --max-len so single-nt is never truncated
    # (that's what makes the two tokenizers residue-matched, not just step-matched)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--max-windows", type=int, default=128, help="per genome (0 = all)")
    p.add_argument("--sampling", choices=["even", "head"], default="even",
                   help="even = windows spread across the whole genome (default)")
    p.add_argument("--max-train-windows", type=int, default=40_000,
                   help="cap on windows used to train tokenizer+LM (0 = all)")
    # LM
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    # protocol
    p.add_argument("--train-split", choices=["train", "all"], default="train",
                   help="pretrain tokenizer+LM only on the train split (default, "
                        "no test peeking) or on all genomes")
    p.add_argument("--split-level", choices=["species", "genus", "family"], default="family")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.window > args.max_len:
        print(
            f"WARNING: --window ({args.window}) > --max-len ({args.max_len}). The "
            "single-nt model will be truncated to max-len tokens, so it sees FEWER "
            "nucleotides per window than domain-BPE — the comparison is no longer "
            "residue-matched. Set --window <= --max-len for a fair run.",
            file=sys.stderr,
        )

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    print(f"device: {device}")

    manifest = CorpusManifest.load(args.data_dir)
    n_ok = len(manifest.ok)
    print(f"corpus: {n_ok} genomes")
    if n_ok == 0:
        sys.exit("empty corpus — run build_genome_corpus.py first")

    split = "train" if args.train_split == "train" else None
    train_pool, per_genome = gather_windows(
        manifest,
        window=args.window, stride=args.stride,
        max_windows_per_genome=(args.max_windows or None),
        split=split, split_level=args.split_level, sampling=args.sampling,
    )
    if args.max_train_windows and len(train_pool) > args.max_train_windows:
        random.shuffle(train_pool)
        train_pool = train_pool[: args.max_train_windows]
    print(f"training pool: {len(train_pool):,} windows ({args.window}bp); "
          f"{len(per_genome)} genomes to embed")

    # 1. tokenizer
    if args.tokenizer == "single_nt":
        tokenizer = NucleotideTokenizer()
        tok_name = "single_nt"
    else:
        print(f"training domain BPE (vocab={args.bpe_vocab}) ...")
        tokenizer = DomainBPETrainer(vocab_size=args.bpe_vocab).train_on_sequences(
            train_pool, name=f"domain_bpe_{args.bpe_vocab}"
        )
        tok_name = tokenizer.name
        tok_dir = args.data_dir / "tokenizers" / tok_name
        tokenizer.save(tok_dir)
        print(f"  saved tokenizer -> {tok_dir}  (vocab_size={tokenizer.vocab_size})")

    # 2. train LM
    cfg = LMConfig(d_model=args.d_model, n_heads=args.n_heads,
                   n_layers=args.n_layers, max_len=args.max_len)
    print(f"training TinyGPT ({tok_name}) ...")
    model, stats = train_lm(
        train_pool, tokenizer, cfg,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, device=device,
    )
    print(f"  LM: {stats['params']:,} params, final_loss={stats['final_loss']}")

    bpr = eval_bits_per_residue(model, tokenizer, train_pool[:2000], device=device)
    print(f"  bits/residue (train-pool diagnostic): {bpr:.4f}")

    # 3. extract per-genome features
    print("extracting genome embeddings ...")
    ids: list[int] = []
    feats: list[np.ndarray] = []
    start = time.time()
    for i, (bid, wins) in enumerate(per_genome, start=1):
        vec = genome_embedding(model, tokenizer, wins, device=device)
        ids.append(bid)
        feats.append(vec)
        if i % 200 == 0:
            print(f"  [{i}/{len(per_genome)}] {i/max(time.time()-start,1e-6):.1f}/s", flush=True)

    out = args.out or (args.data_dir / f"{tok_name}_features.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        bacdive_ids=np.array(ids, dtype=np.int64),
        features=np.stack(feats).astype(np.float32),
    )
    meta = {
        "tokenizer": tok_name,
        "vocab_size": int(tokenizer.vocab_size),
        "feature_dim": int(cfg.d_model),
        "n_genomes": len(ids),
        "window": args.window, "stride": args.stride, "max_windows": args.max_windows,
        "sampling": args.sampling,
        "lm": stats, "bits_per_residue_diag": round(float(bpr), 4),
        "train_split": args.train_split, "split_level": args.split_level,
        # fairness bookkeeping: with window<=max_len both tokenizers see the same
        # nucleotides over the same #steps, so they are residue-matched.
        "residue_matched": bool(args.window <= args.max_len),
        "approx_residues_seen": int(args.steps * args.batch_size * args.window),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}  [{len(ids)}, {cfg.d_model}]")
    print(f"wrote {out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
