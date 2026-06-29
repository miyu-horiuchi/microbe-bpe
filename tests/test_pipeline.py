"""CPU-only unit tests: tokenizers, windowing, and the TinyGPT feature path.

No network and no microbe-foundation checkout required.
"""

from __future__ import annotations

import random

import numpy as np

from microbe_bpe.genome_corpus import contigs_to_genome, normalize_dna, window_dna
from microbe_bpe.tiny_lm import LMConfig, genome_embedding, train_lm
from microbe_bpe.tokenizers import DomainBPETrainer, NucleotideTokenizer


def _random_dna(n: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    # motif-biased so BPE has structure to find
    motifs = ["ATGAAA", "GGGGCC", "TTTAAA", "GAATTC"]
    out = []
    while sum(len(x) for x in out) < n:
        if rng.random() < 0.4:
            out.append(rng.choice(motifs))
        else:
            out.append("".join(rng.choice("ACGT") for _ in range(rng.randint(3, 9))))
    return "".join(out)[:n]


def test_normalize_and_concat():
    contigs = [("c1", "acgtN"), ("c2", "ACGTX")]
    assert normalize_dna("acgtxn") == "ACGTNN"
    genome = contigs_to_genome(contigs, cap=None)
    assert set(genome) <= set("ACGTN")
    assert len(genome) == 10
    assert contigs_to_genome(contigs, cap=4) == genome[:4]


def test_window_dna():
    dna = "ACGT" * 100  # 400 bp
    wins = window_dna(dna, window=100, stride=50, max_windows=None)
    assert all(len(w) == 100 for w in wins)
    assert len(wins) == (400 - 100) // 50 + 1
    assert window_dna(dna, 100, 50, max_windows=3) == wins[:3]
    assert window_dna("ACGTACGT", 100, 50, None) == ["ACGTACGT"]  # short genome
    assert window_dna("", 100, 50, None) == []


def test_nucleotide_tokenizer_roundtrip():
    tok = NucleotideTokenizer()
    assert tok.vocab_size == 4 + 5  # SPECIAL(4) + ACGTN(5)
    ids = tok.encode("ACGTN")
    assert len(ids) == 5
    assert tok.decode(ids) == "ACGTN"


def test_domain_bpe_learns_merges():
    seqs = [_random_dna(500, seed=i) for i in range(40)]
    tok = DomainBPETrainer(vocab_size=300).train_on_sequences(seqs, name="domain_bpe_300")
    assert tok.vocab_size > 9  # learned merges beyond single nucleotides + specials
    # BPE should compress motif-rich DNA below 1 token/nucleotide
    sample = seqs[0]
    assert len(tok.encode(sample)) < len(sample)


def test_tinygpt_train_and_embed():
    seqs = [_random_dna(400, seed=i) for i in range(60)]
    windows = [w for s in seqs for w in window_dna(s, 200, 200, None)]
    tok = NucleotideTokenizer()
    cfg = LMConfig(d_model=32, n_heads=2, n_layers=1, max_len=128)
    model, stats = train_lm(windows, tok, cfg, steps=20, batch_size=16, verbose=False)
    assert stats["params"] > 0
    vec = genome_embedding(model, tok, window_dna(seqs[0], 200, 200, None))
    assert vec.shape == (32,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()
    # empty genome -> zero vector of right size
    assert genome_embedding(model, tok, []).shape == (32,)


def test_kmer_mock_vector():
    from extract_evo2_features import _kmer_vector

    v = _kmer_vector("ACGT" * 50, k=4)
    assert v.shape == (256,)
    assert abs(float(v.sum()) - 1.0) < 1e-5  # normalized frequencies
