"""Genome tokenizers: single-nucleotide baseline + domain-adaptive BPE.

Adapted for DNA from miyu-horiuchi/BPE (bpe/tokenizers.py). Two tokenizers,
identical interface (encode/decode/tokenize/vocab_size), so the same TinyGPT
trains under each and only the tokenization changes:

  NucleotideTokenizer  one token per nucleotide (A/C/G/T/N) — the "single
                       residue" baseline whose token distribution is just the
                       marginal nucleotide frequencies (the tokenization trap).

  domain BPE           byte-level BPE merges learned directly on microbial DNA,
                       so frequent k-mers/motifs become single tokens and the
                       token distribution moves into the language-like band.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from . import DNA_ALPHABET_N

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


class SequenceTokenizer(Protocol):
    name: str

    def encode(self, sequence: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def tokenize(self, sequence: str) -> list[str]: ...


@dataclass
class NucleotideTokenizer:
    """One token per nucleotide — the single-residue baseline.

    Token id 0 is <pad> (matches TinyGPT's pad_id), so a genome of length L
    encodes to L token ids.
    """

    name: str = "single_nt"

    def __post_init__(self) -> None:
        self._stoi = {nt: i + len(SPECIAL) for i, nt in enumerate(DNA_ALPHABET_N)}
        self._itos = {i: nt for nt, i in self._stoi.items()}
        for i, tok in enumerate(SPECIAL):
            self._itos[i] = tok

    @property
    def vocab_size(self) -> int:
        return len(SPECIAL) + len(DNA_ALPHABET_N)

    def tokenize(self, sequence: str) -> list[str]:
        return [c if c in self._stoi else "<unk>" for c in sequence.upper()]

    def encode(self, sequence: str) -> list[int]:
        return [self._stoi.get(c, self._stoi["N"]) for c in sequence.upper() if c.isalpha()]

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self._itos.get(i, "") for i in ids if self._itos.get(i, "") not in SPECIAL
        )


@dataclass
class HuggingFaceBPETokenizer:
    """Wrapper around a trained/loaded HF `tokenizers` BPE model."""

    tokenizer: Tokenizer
    name: str

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def tokenize(self, sequence: str) -> list[str]:
        return self.tokenizer.encode(sequence).tokens

    def encode(self, sequence: str) -> list[int]:
        return self.tokenizer.encode(sequence).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(directory / "tokenizer.json"))
        (directory / "meta.json").write_text(json.dumps({"name": self.name}))

    @staticmethod
    def load(directory: Path) -> "HuggingFaceBPETokenizer":
        meta = json.loads((directory / "meta.json").read_text())
        tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
        return HuggingFaceBPETokenizer(tokenizer=tok, name=meta["name"])


class DomainBPETrainer:
    """Train byte-level BPE on DNA strings (no spaces between nucleotides)."""

    def __init__(self, vocab_size: int = 1024, min_frequency: int = 2) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def train_on_sequences(
        self, sequences: Iterable[str], name: str = "domain_bpe"
    ) -> HuggingFaceBPETokenizer:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            for s in sequences:
                if s:
                    fh.write(s + "\n")
            corpus_path = Path(fh.name)
        try:
            return self.train(corpus_path, name=name)
        finally:
            corpus_path.unlink(missing_ok=True)

    def train(self, corpus_path: Path, name: str = "domain_bpe") -> HuggingFaceBPETokenizer:
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        # ByteLevel on ASCII ACGTN strings == char-level BPE with working merge stats.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=max(1, self.min_frequency),
            special_tokens=SPECIAL,
            show_progress=False,
        )
        tokenizer.train([str(corpus_path)], trainer)
        return HuggingFaceBPETokenizer(tokenizer=tokenizer, name=name)


def load_tokenizer(kind: str, artifacts_dir: Path | None = None) -> SequenceTokenizer:
    """Load a tokenizer by kind: 'single_nt' or a saved domain-BPE directory name."""
    if kind == "single_nt":
        return NucleotideTokenizer()
    if artifacts_dir is None:
        raise ValueError(f"artifacts_dir required to load BPE tokenizer {kind!r}")
    path = artifacts_dir / kind
    if path.exists():
        return HuggingFaceBPETokenizer.load(path)
    raise FileNotFoundError(f"tokenizer {kind!r} not found under {artifacts_dir}")
