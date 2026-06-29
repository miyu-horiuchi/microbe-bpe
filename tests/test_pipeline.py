"""CPU-only unit tests: tokenizers, windowing, and the TinyGPT feature path.

No network and no microbe-foundation checkout required.
"""

from __future__ import annotations

import random

import numpy as np

from microbe_bpe.genome_corpus import contigs_to_genome, normalize_dna, window_dna
from microbe_bpe.intrinsic import compression_ratio, gc_content, token_counts, zipf_exponent
from microbe_bpe.tiny_lm import LMConfig, genome_embedding, train_lm
from microbe_bpe.tokenizers import DomainBPETrainer, KmerTokenizer, NucleotideTokenizer


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
    rng = random.Random(0)
    dna = "".join(rng.choice("ACGT") for _ in range(400))  # non-periodic, distinct windows
    wins = window_dna(dna, window=100, stride=50, max_windows=None)
    assert all(len(w) == 100 for w in wins)
    assert len(wins) == (400 - 100) // 50 + 1  # starts 0,50,...,300 -> 7 windows
    # legacy "head" sampling = first N windows (prefix)
    assert window_dna(dna, 100, 50, max_windows=3, sampling="head") == wins[:3]
    # default "even" sampling spreads across the WHOLE genome: starts 0,150,300
    even = window_dna(dna, 100, 50, max_windows=3)
    assert len(even) == 3
    assert even[0] == dna[0:100]
    assert even[1] == dna[150:250]
    assert even[2] == dna[300:400]
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


def test_bpe_pool_aggregates_by_token():
    from microbe_bpe.tiny_lm import bpe_pool

    # per-position embeddings (L=4, D=1): rows 0,2,4,6
    positions = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    # two BPE tokens spanning bases [0:2) and [2:4)
    offsets = [(0, 2), (2, 4)]
    out = bpe_pool(positions, offsets)
    # token means: (0+2)/2=1, (4+6)/2=5 ; mean over tokens = 3
    assert out.shape == (1,)
    assert abs(float(out[0]) - 3.0) < 1e-6
    # out-of-range spans are clamped; empty offsets fall back to global mean
    assert abs(float(bpe_pool(positions, [])[0]) - 3.0) < 1e-6


def test_bpe_offsets_cover_sequence():
    seqs = [_random_dna(400, seed=i) for i in range(30)]
    tok = DomainBPETrainer(vocab_size=200).train_on_sequences(seqs, name="domain_bpe_200")
    seq = seqs[0]
    ids, offsets = tok.encode_with_offsets(seq)
    assert len(ids) == len(offsets)
    # byte-level offsets are contiguous and cover the whole sequence
    assert offsets[0][0] == 0
    assert offsets[-1][1] == len(seq)
    for (s, e), (ns, _ne) in zip(offsets, offsets[1:]):
        assert e == ns  # contiguous


def test_bpe_bag_mock_vector():
    from extract_evo2_features import _bpe_bag_vector

    seqs = [_random_dna(300, seed=i) for i in range(20)]
    tok = DomainBPETrainer(vocab_size=128).train_on_sequences(seqs, name="domain_bpe_128")
    v = _bpe_bag_vector(seqs[0], tok)
    assert v.shape == (tok.vocab_size,)
    assert abs(float(v.sum()) - 1.0) < 1e-5


def test_kmer_tokenizer():
    tok = KmerTokenizer(k=3)
    assert tok.name == "kmer_3"
    assert tok.vocab_size == 4 + 4 ** 3  # SPECIAL + 4^k
    ids = tok.encode("ACGTAA")  # two clean 3-mers: ACG, TAA
    assert len(ids) == 2
    # trailing remainder shorter than k is dropped
    assert len(tok.encode("ACGTA")) == 1
    # offsets are fixed k-base spans, contiguous from 0
    ids2, offs = tok.encode_with_offsets("ACGTAA")
    assert offs == [(0, 3), (3, 6)]
    # non-ACGT chunk -> <unk>, but still one token
    assert len(tok.encode("ACN")) == 1
    # same chunk -> same id (deterministic)
    assert tok.encode("ACGACG") == [tok.encode("ACG")[0]] * 2


def test_zipf_and_compression():
    seqs = [_random_dna(400, seed=i) for i in range(40)]
    nt = NucleotideTokenizer()
    bpe = DomainBPETrainer(vocab_size=256).train_on_sequences(seqs, name="domain_bpe_256")
    # single-nt compresses 1 nt/token; BPE should merge -> > 1
    assert abs(compression_ratio(nt, seqs) - 1.0) < 1e-9
    assert compression_ratio(bpe, seqs) > 1.0
    # single-nt token stats are ~flat (only 4 symbols); BPE is more Zipfian
    z_nt = zipf_exponent(token_counts(nt, seqs))
    z_bpe = zipf_exponent(token_counts(bpe, seqs))
    assert z_bpe > z_nt
    # zipf needs >= 3 distinct tokens
    assert np.isnan(zipf_exponent({1: 10, 2: 5}))


def test_gc_content():
    assert abs(gc_content("GCGC") - 1.0) < 1e-9
    assert abs(gc_content("ATAT") - 0.0) < 1e-9
    assert abs(gc_content("ACGT") - 0.5) < 1e-9
    assert abs(gc_content("GGCCATNN") - 4 / 6) < 1e-9  # N ignored
