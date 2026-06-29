"""Shared genome-DNA corpus: fetch once, both tokenizers consume the same input.

The whole point of the comparison is that Evo2 and the domain-BPE genome model
see *identical* nucleotide input, so any difference is attributable to the
tokenizer/representation and not to data. This module owns:

  - fetching genome contigs from NCBI (via microbe-foundation's fetcher),
  - normalizing to an uppercase ACGTN string (one per genome, optionally capped),
  - caching to data/genome_dna/<bacdive_id>.txt.gz + a manifest parquet,
  - a windowing helper used by both feature extractors.

The cache is the single source of truth; extractors never re-fetch.
"""

from __future__ import annotations

import gzip
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import DNA_ALPHABET_N
from .mf_bridge import fetch_contigs

_VALID = set(DNA_ALPHABET_N)


def normalize_dna(seq: str) -> str:
    """Uppercase; map anything outside ACGT to N (keeps length, kills ambiguity)."""
    seq = seq.upper()
    return "".join(c if c in _VALID else "N" for c in seq)


def contigs_to_genome(contigs: list[tuple[str, str]], cap: int | None) -> str:
    """Concatenate contigs into one normalized DNA string, optionally length-capped.

    Contigs are joined directly (no spacer) so BPE merge statistics are not
    polluted by an artificial separator token; the cap keeps dev runs cheap.
    """
    genome = "".join(normalize_dna(seq) for _, seq in contigs)
    if cap and len(genome) > cap:
        genome = genome[:cap]
    return genome


def window_dna(dna: str, window: int, stride: int, max_windows: int | None) -> list[str]:
    """Slice a genome into overlapping windows. Used by both extractors."""
    if not dna:
        return []
    if len(dna) <= window:
        return [dna]
    out: list[str] = []
    for i in range(0, len(dna) - window + 1, stride):
        out.append(dna[i : i + window])
        if max_windows and len(out) >= max_windows:
            break
    return out


# --------------------------------------------------------------------------- #
# Cache layout
# --------------------------------------------------------------------------- #

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DNA_SUBDIR = "genome_dna"
MANIFEST_NAME = "genome_manifest.parquet"


def dna_path(data_dir: Path, bacdive_id: int) -> Path:
    return data_dir / DNA_SUBDIR / f"{bacdive_id}.txt.gz"


def write_dna(path: Path, dna: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write(dna)


def read_dna(path: Path) -> str:
    with gzip.open(path, "rt") as fh:
        return fh.read()


@dataclass
class CorpusManifest:
    """Index of the cached genome DNA, with split labels joined in if available."""

    df: pd.DataFrame  # columns: bacdive_id, accession, length, path, status[, split]

    @property
    def ok(self) -> pd.DataFrame:
        return self.df[self.df.status == "ok"].reset_index(drop=True)

    def iter_dna(self, data_dir: Path, split: str | None = None):
        """Yield (bacdive_id, dna) for cached genomes, optionally filtered by split."""
        sub = self.ok
        if split is not None and "split" in sub.columns:
            sub = sub[sub.split == split]
        for row in sub.itertuples():
            yield int(row.bacdive_id), read_dna(Path(row.path))

    def save(self, data_dir: Path) -> Path:
        out = data_dir / MANIFEST_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(out, index=False)
        return out

    @classmethod
    def load(cls, data_dir: Path) -> "CorpusManifest":
        path = data_dir / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"No genome manifest at {path}. Run build_genome_corpus.py first."
            )
        return cls(pd.read_parquet(path))


def _attach_splits(df: pd.DataFrame, mf_root: str | None) -> pd.DataFrame:
    """Best-effort join of species/genus/family split labels from microbe-foundation."""
    try:
        from .mf_bridge import mf_data_dir

        splits_path = mf_data_dir(mf_root) / "splits.parquet"
        if not splits_path.exists():
            return df
        splits = pd.read_parquet(splits_path)
        keep = ["bacdive_id"] + [c for c in splits.columns if c.endswith("_split")]
        return df.merge(splits[keep], on="bacdive_id", how="left")
    except Exception:
        return df


def build_corpus(
    accessions: pd.DataFrame,
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    cap: int | None = 200_000,
    limit: int | None = None,
    mf_root: str | None = None,
    checkpoint_every: int = 25,
    sleep: float = 0.0,
) -> CorpusManifest:
    """Fetch + cache genome DNA for each (bacdive_id, accession) row.

    Resumable: rows whose DNA file already exists are skipped. `cap` bounds the
    per-genome length so dev runs stay cheap; pass cap=None for full genomes.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    if limit:
        accessions = accessions.head(limit)

    rows: list[dict] = []
    n_ok = n_fail = 0
    start = time.time()
    for i, row in enumerate(accessions.itertuples(), start=1):
        bid = int(row.bacdive_id)
        acc = str(row.accession)
        path = dna_path(data_dir, bid)

        if path.exists():
            try:
                length = len(read_dna(path))
                rows.append({"bacdive_id": bid, "accession": acc, "length": length,
                             "path": str(path), "status": "ok"})
                n_ok += 1
                continue
            except OSError:
                pass  # corrupt cache -> re-fetch

        try:
            contigs = fetch_contigs(acc, mf_root)
        except Exception as e:  # noqa: BLE001 - network/parse failures are expected
            print(f"  [warn] {bid} ({acc}) fetch failed: {type(e).__name__}: {e}", flush=True)
            contigs = None

        if not contigs:
            rows.append({"bacdive_id": bid, "accession": acc, "length": 0,
                         "path": "", "status": "fetch_fail"})
            n_fail += 1
        else:
            dna = contigs_to_genome(contigs, cap)
            write_dna(path, dna)
            rows.append({"bacdive_id": bid, "accession": acc, "length": len(dna),
                         "path": str(path), "status": "ok"})
            n_ok += 1

        if i % checkpoint_every == 0:
            man = CorpusManifest(_attach_splits(pd.DataFrame(rows), mf_root))
            man.save(data_dir)
            rate = i / max(time.time() - start, 1e-6)
            print(f"  [{i:>5}/{len(accessions)}] ok={n_ok} fail={n_fail} "
                  f"rate={rate:.2f}/s", flush=True)
        if sleep:
            time.sleep(sleep)

    manifest = CorpusManifest(_attach_splits(pd.DataFrame(rows), mf_root))
    manifest.save(data_dir)
    print(f"\ncorpus: {n_ok} cached, {n_fail} failed -> {data_dir / MANIFEST_NAME}")
    return manifest
