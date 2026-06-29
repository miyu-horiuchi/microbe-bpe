#!/usr/bin/env python3
"""Intrinsic report — the tokenization-trap claim tested WITHOUT a GPU or labels.

Reads the per-method *_features.meta.json (written by extract_bpe_features.py) and
the cached genome DNA, and writes results/intrinsic.md with:

  1. Intrinsic LM table — vocab, nt/token (compression), Zipf exponent, and
     held-out bits-per-residue per method. This is what the paper actually argues:
     domain-BPE should compress better (lower bpr) with more language-like
     (higher Zipf) token statistics than the single-nucleotide baseline, with
     fixed k-mers somewhere in between.

  2. GC-content positive-control probe — ridge regression (CV R^2) predicting each
     genome's GC content from its frozen features. GC is trivially DNA-decodable,
     so it's a sanity check that signal exists: if a representation can't predict
     GC, its downstream trait nulls are uninformative.

Usage:
    python report_intrinsic.py            # scans data/*_features*.npz + .meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from microbe_bpe.genome_corpus import DEFAULT_DATA_DIR, CorpusManifest, read_dna
from microbe_bpe.intrinsic import gc_content

REPO = Path(__file__).resolve().parent
RESULTS_DIR = REPO / "results"


def gc_by_id(data_dir: Path) -> dict[int, float]:
    """Map bacdive_id -> GC content, computed from the cached genome DNA."""
    try:
        manifest = CorpusManifest.load(data_dir)
    except Exception:
        return {}
    out: dict[int, float] = {}
    for row in manifest.ok.itertuples():
        try:
            out[int(row.bacdive_id)] = gc_content(read_dna(Path(row.path)))
        except Exception:
            continue
    return out


def gc_probe_r2(ids: np.ndarray, feats: np.ndarray, gc_map: dict[int, float]):
    """Cross-validated R^2 predicting GC from features (None if not enough data)."""
    y = np.array([gc_map.get(int(i), np.nan) for i in ids], dtype=np.float64)
    keep = np.isfinite(y)
    X, y = feats[keep], y[keep]
    n = len(y)
    # need enough genomes that each CV test fold has >= 2 points, else R^2 is degenerate
    if n < 8 or np.std(y) < 1e-9:
        return None, n
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None, n
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    splits = max(2, min(5, n // 2))
    scores = cross_val_score(model, X, y, cv=KFold(n_splits=splits, shuffle=True, random_state=0),
                             scoring="r2")
    val = float(np.nanmean(scores))
    return (val if np.isfinite(val) else None), n


def discover(data_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for npz in sorted(data_dir.glob("*features*.npz")):
        name = npz.name
        for suffix in (".MOCK.npz", "_features.npz", ".npz"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        name = name.replace("_features", "").strip("._") or npz.stem
        if npz.name.endswith(".MOCK.npz"):
            name += "_MOCK"
        out[name] = npz
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = p.parse_args()

    features = discover(args.data_dir)
    if not features:
        raise SystemExit(f"no *_features*.npz in {args.data_dir}; run the extractors first")

    gc_map = gc_by_id(args.data_dir)

    lines = ["# Intrinsic report — tokenization-trap diagnostics (no GPU, no labels)", ""]
    lines.append("Lower **bits/residue** = better compression. Higher **Zipf exp** "
                 "(~1.0) = more language-like token stats. **nt/token** = mean "
                 "nucleotides per token (single-nt=1; k-mer=k; domain-BPE variable). "
                 "**GC R²** is a positive-control probe (cross-validated).")
    lines.append("")
    lines.append("| Method | vocab | nt/token | Zipf exp | bits/residue | bpr held-out? | GC R² (n) |")
    lines.append("|---|---:|---:|---:|---:|:--:|---:|")

    for name, npz in features.items():
        meta_path = npz.with_suffix(".meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                meta = {}
        d = np.load(npz)
        gc_r2, n = gc_probe_r2(d["bacdive_ids"], d["features"].astype(np.float64), gc_map)

        def g(k, fmt="{}"):
            v = meta.get(k)
            return fmt.format(v) if v is not None else "—"

        vocab = g("vocab_size")
        ntt = g("nt_per_token", "{:.2f}")
        zipf = g("zipf_exponent", "{:.3f}")
        bpr = g("bits_per_residue", "{:.4f}")
        held = "yes" if meta.get("bits_per_residue_held_out") else ("no" if "bits_per_residue" in meta else "—")
        gc_s = f"{gc_r2:+.3f} ({n})" if gc_r2 is not None else f"— ({n})"
        lines.append(f"| `{name}` | {vocab} | {ntt} | {zipf} | {bpr} | {held} | {gc_s} |")

    lines += [
        "",
        "## How to read this",
        "- **The paper's core prediction:** `single_nt` should have ~flat token stats "
        "(low Zipf) and worse bits/residue than `domain_bpe`; `kmer` sits between. "
        "Intrinsic metrics are GPU-free and confound-free — this is the cleanest test.",
        "- **GC R²** only sanity-checks that a representation carries decodable signal. "
        "Mock Evo2 features are k-mer/bag stand-ins, so a high GC R² there is expected "
        "and NOT evidence about real Evo2.",
        "- Trait prediction (results/comparison.md) is the *secondary*, harder endpoint.",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "intrinsic.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}\n")
    print("\n".join(lines[:8 + len(features)]))


if __name__ == "__main__":
    main()
