"""Locate and reuse a microbe-foundation checkout.

microbe-bpe is an *extension* of microbe-foundation: it reuses that repo's
benchmark data (traits/splits/vocab/schema), its NCBI genome fetch, its
multi-task model (`model.py`), and its `leaderboard.py`. Rather than vendor and
drift from ~50 files, we point at a sibling checkout.

Resolution order for the microbe-foundation root:
    1. --mf-root CLI arg (passed through to resolve_mf_root)
    2. $MF_ROOT environment variable
    3. ./microbe-foundation git submodule (the default, vendored in this repo)
    4. ../microbe-foundation next to this repo
    5. ~/Documents/GitHub/microbe-foundation

The submodule ships with the repo; `git submodule update --init` (or
`scripts/setup.sh`) populates it. The sibling/home fallbacks remain so an
existing local checkout still works.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Files we require to exist to consider a directory a valid microbe-foundation.
_SENTINELS = ("model.py", "leaderboard.py", "trait_schema.json", "microbe_model")


def _looks_like_mf(path: Path) -> bool:
    return path.is_dir() and all((path / s).exists() for s in _SENTINELS)


def _candidates(explicit: str | os.PathLike[str] | None) -> list[Path]:
    out: list[Path] = []
    if explicit:
        out.append(Path(explicit).expanduser())
    env = os.environ.get("MF_ROOT")
    if env:
        out.append(Path(env).expanduser())
    out.append(_REPO_ROOT / "microbe-foundation")          # vendored submodule
    out.append(_REPO_ROOT.parent / "microbe-foundation")   # sibling checkout
    out.append(Path.home() / "Documents" / "GitHub" / "microbe-foundation")
    return out


@lru_cache(maxsize=8)
def resolve_mf_root(explicit: str | None = None) -> Path:
    """Return the microbe-foundation repo root, or raise with guidance."""
    for cand in _candidates(explicit):
        if _looks_like_mf(cand):
            return cand.resolve()
    searched = "\n  ".join(str(c) for c in _candidates(explicit))
    raise FileNotFoundError(
        "Could not find a microbe-foundation checkout. Looked in:\n  "
        f"{searched}\n\n"
        "Fix: clone it next to this repo and/or set MF_ROOT, e.g.\n"
        "  git clone https://github.com/miyu-horiuchi/microbe-foundation "
        "../microbe-foundation\n"
        "or:  export MF_ROOT=/path/to/microbe-foundation\n"
        "(scripts/setup.sh does this for you.)"
    )


def add_mf_to_path(explicit: str | None = None) -> Path:
    """Put the microbe-foundation root on sys.path so `microbe_model` imports.

    Returns the resolved root.
    """
    root = resolve_mf_root(explicit)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def mf_data_dir(explicit: str | None = None) -> Path:
    return resolve_mf_root(explicit) / "data"


def fetch_contigs(accession: str, explicit_mf_root: str | None = None):
    """Fetch genome contigs [(name, seq), ...] from NCBI Datasets, in memory.

    Prefers microbe-foundation's fetcher when its package imports cleanly (so the
    DNA matches that pipeline exactly); otherwise uses a self-contained copy of
    the same NCBI Datasets logic. The fallback means the DNA-only corpus build
    does not require microbe-foundation's heavier deps (pyrodigal/biopython).
    """
    try:
        add_mf_to_path(explicit_mf_root)
        from microbe_model.pipeline import _fetch_fasta_bytes  # type: ignore

        return _fetch_fasta_bytes(accession)
    except (FileNotFoundError, ImportError):
        return _fetch_fasta_bytes_standalone(accession)


# --------------------------------------------------------------------------- #
# Self-contained NCBI Datasets fetch (mirrors microbe_model.pipeline)
# --------------------------------------------------------------------------- #

_DATASETS_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/download"
_VERSION_FALLBACKS = (".1", ".2", ".3", ".4")
_EMPTY_ZIP_BYTES = 2_000


def _has_version(accession: str) -> bool:
    if "." not in accession:
        return False
    return accession.rsplit(".", 1)[-1].isdigit()


def _candidate_accessions(accession: str) -> list[str]:
    if _has_version(accession):
        return [accession]
    return [accession + v for v in _VERSION_FALLBACKS]


def _parse_fasta_bytes(raw: bytes) -> list[tuple[str, str]]:
    contigs: list[tuple[str, str]] = []
    cur_id: str | None = None
    chunks: list[str] = []
    for line in raw.splitlines():
        if not line:
            continue
        if line.startswith(b">"):
            if cur_id is not None:
                contigs.append((cur_id, "".join(chunks).upper()))
            cur_id = line[1:].decode("ascii", errors="replace").split()[0]
            chunks = []
        else:
            chunks.append(line.decode("ascii", errors="replace"))
    if cur_id is not None:
        contigs.append((cur_id, "".join(chunks).upper()))
    return contigs


def _fetch_fasta_bytes_standalone(accession: str):
    import io
    import time
    import zipfile

    import requests

    api_key = os.environ.get("NCBI_API_KEY")
    headers = {"Accept": "application/zip"}
    if api_key:
        headers["api-key"] = api_key
    params = {"include_annotation_type": "GENOME_FASTA"}
    rate = 0.1 if api_key else 0.34

    for candidate in _candidate_accessions(accession):
        zip_bytes = None
        for attempt in range(3):
            try:
                time.sleep(rate)
                resp = requests.get(_DATASETS_URL.format(acc=candidate),
                                    params=params, headers=headers, timeout=120)
                if resp.status_code == 404:
                    break
                if resp.status_code in (429, 502, 503):
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
            except requests.RequestException:
                if attempt == 2:
                    break
                time.sleep(2 ** attempt)
                continue
            if len(resp.content) >= _EMPTY_ZIP_BYTES:
                zip_bytes = resp.content
            break
        if zip_bytes is None:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                fna = [n for n in zf.namelist() if n.endswith(".fna")]
                if not fna:
                    continue
                raw = zf.open(fna[0]).read()
        except zipfile.BadZipFile:
            continue
        return _parse_fasta_bytes(raw)
    return None
