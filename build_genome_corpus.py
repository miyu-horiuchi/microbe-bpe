#!/usr/bin/env python3
"""Step 1 — build the shared genome-DNA corpus.

Fetches genome contigs for each (bacdive_id, accession) and caches one
normalized DNA string per genome under data/genome_dna/, plus a manifest. Both
the Evo2 and domain-BPE feature extractors read this cache, so they see byte-
identical input (the comparison is then purely about the tokenizer).

Accessions come from microbe-foundation's extract_genome_accessions.py output
(data/genome_accessions.tsv). Build that first in the microbe-foundation repo:

    python fetch_bacdive.py && python parse_bacdive.py && python splits.py \\
        && python vocab.py && python extract_genome_accessions.py

Usage:
    # dev subset (200 genomes, capped at 200 kb each) — minutes, laptop OK
    python build_genome_corpus.py --limit 200

    # full corpus, uncapped genomes
    python build_genome_corpus.py --cap 0

    # tiny self-contained demo (no microbe-foundation data needed; feature
    # pipeline only — these ids won't join model.py's labels)
    python build_genome_corpus.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from microbe_bpe.genome_corpus import DEFAULT_DATA_DIR, build_corpus
from microbe_bpe.mf_bridge import mf_data_dir

# A few small, well-assembled reference genomes for --demo (RefSeq accessions).
DEMO_ACCESSIONS = [
    (900001, "GCF_000005845.2"),  # Escherichia coli K-12 MG1655
    (900002, "GCF_000009045.1"),  # Bacillus subtilis 168
    (900003, "GCF_000006765.1"),  # Pseudomonas aeruginosa PAO1
    (900004, "GCF_000195955.2"),  # Mycobacterium tuberculosis H37Rv
    (900005, "GCF_000013425.1"),  # Staphylococcus aureus NCTC 8325
    (900006, "GCF_000008865.2"),  # Escherichia coli O157:H7 Sakai
]


def load_accessions(args: argparse.Namespace) -> pd.DataFrame:
    if args.demo:
        print("demo mode: using a small built-in accession list")
        return pd.DataFrame(DEMO_ACCESSIONS, columns=["bacdive_id", "accession"])

    path = args.accessions
    if path is None:
        try:
            path = mf_data_dir(args.mf_root) / "genome_accessions.tsv"
        except FileNotFoundError as e:
            sys.exit(str(e))
    path = Path(path)
    if not path.exists():
        sys.exit(
            f"Accessions file not found: {path}\n"
            "Run microbe-foundation's extract_genome_accessions.py first, or pass "
            "--accessions / --demo."
        )
    df = pd.read_csv(path, sep="\t")
    if not {"bacdive_id", "accession"}.issubset(df.columns):
        sys.exit(f"{path} must have columns: bacdive_id, accession (got {list(df.columns)})")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accessions", type=Path, default=None,
                   help="TSV with bacdive_id, accession (default: microbe-foundation's)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--mf-root", type=str, default=None, help="microbe-foundation checkout path")
    p.add_argument("--cap", type=int, default=200_000,
                   help="Max nucleotides cached per genome (0 = uncapped). Default 200kb.")
    p.add_argument("--limit", type=int, default=0, help="Process at most N genomes (0 = all)")
    p.add_argument("--demo", action="store_true", help="Use built-in demo accessions")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between fetches")
    args = p.parse_args()

    accessions = load_accessions(args)
    print(f"{len(accessions):,} accessions to consider")
    build_corpus(
        accessions,
        data_dir=args.data_dir,
        cap=(args.cap or None),
        limit=(args.limit or None),
        mf_root=args.mf_root,
        sleep=args.sleep,
    )


if __name__ == "__main__":
    main()
