"""Intrinsic, tokenizer-agnostic diagnostics — the metrics the tokenization-trap
paper actually argues about (no trait labels, no GPU needed).

  * compression (nucleotides per token): single-nt = 1.0, kmer = k, domain-BPE
    is variable (> 1 if it really merges frequent motifs).
  * Zipf exponent: slope of log(frequency) vs log(rank) over the token
    distribution. "Language-like" corpora sit near ~1.0; the single-residue
    baseline is flat (just the 4 nucleotide marginals) — the "trap".
  * GC content: a trivially DNA-decodable property used as a POSITIVE-CONTROL
    probe target (if a representation can't linearly predict GC, downstream
    trait nulls are uninformative).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

SPECIAL_TOKENS = {"<pad>", "<bos>", "<eos>", "<unk>"}


def token_counts(tokenizer, sequences) -> Counter:
    """Count token occurrences over an iterable of sequences (specials excluded)."""
    counts: Counter = Counter()
    special_ids = set()
    itos = getattr(tokenizer, "_itos", None)
    if itos:
        special_ids = {i for i, t in itos.items() if t in SPECIAL_TOKENS}
    for s in sequences:
        if not s:
            continue
        for tid in tokenizer.encode(s):
            if tid not in special_ids:
                counts[tid] += 1
    return counts


def zipf_exponent(counts) -> float:
    """Least-squares slope magnitude of log10(freq) vs log10(rank).

    Returns a positive number; ~1.0 is Zipfian/"language-like", ~0 is flat.
    Needs at least a few distinct tokens to be meaningful.
    """
    freqs = np.array(sorted((c for c in (counts.values() if isinstance(counts, dict)
                                          else counts) if c > 0), reverse=True),
                     dtype=np.float64)
    if freqs.size < 3:
        return float("nan")
    ranks = np.arange(1, freqs.size + 1, dtype=np.float64)
    slope = np.polyfit(np.log10(ranks), np.log10(freqs), 1)[0]
    return float(-slope)


def compression_ratio(tokenizer, sequences) -> float:
    """Mean nucleotides per token over the sequences (>= 1; higher = more merging)."""
    n_res = 0
    n_tok = 0
    special_ids = set()
    itos = getattr(tokenizer, "_itos", None)
    if itos:
        special_ids = {i for i, t in itos.items() if t in SPECIAL_TOKENS}
    for s in sequences:
        if not s:
            continue
        ids = [t for t in tokenizer.encode(s) if t not in special_ids]
        n_res += len(s)
        n_tok += len(ids)
    return float(n_res / n_tok) if n_tok else float("nan")


def gc_content(dna: str) -> float:
    """Fraction of A/C/G/T bases that are G or C (N and others ignored)."""
    g = dna.count("G") + dna.count("g")
    c = dna.count("C") + dna.count("c")
    a = dna.count("A") + dna.count("a")
    t = dna.count("T") + dna.count("t")
    denom = a + c + g + t
    return float((g + c) / denom) if denom else float("nan")
