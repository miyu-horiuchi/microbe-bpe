#!/usr/bin/env python3
"""Step 4 — run the downstream comparison and summarize.

For each feature set (Evo2, domain-BPE, single-nt) x split (species/genus/family)
x seed, this drives microbe-foundation's model.py to train the 21-head trait
model on those frozen genome features, saving per-run metrics. It then calls
microbe-foundation's leaderboard.py and writes a focused head-to-head summary:
Evo2 (single-nucleotide gLM) vs domain-BPE (the tokenization-trap method).

Usage (after build_genome_corpus.py + the two extractors):
    python run_comparison.py --epochs 30 --splits species genus family --seeds 0 1 2

Auto-discovers data/*_features*.npz unless you pass explicit --features name=path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from microbe_bpe.mf_bridge import resolve_mf_root

REPO = Path(__file__).resolve().parent
RUNS_DIR = REPO / "runs"
RESULTS_DIR = REPO / "results"


def discover_features(data_dir: Path) -> dict[str, Path]:
    """Map a friendly method name -> npz path for every *_features*.npz found."""
    out: dict[str, Path] = {}
    for npz in sorted(data_dir.glob("*features*.npz")):
        stem = npz.name
        for suffix in (".MOCK.npz", "_features.npz", ".npz"):
            if stem.endswith(suffix):
                name = stem[: -len(suffix)] or npz.stem
                break
        else:
            name = npz.stem
        name = name.replace("_features", "").strip("._") or npz.stem
        if npz.name.endswith(".MOCK.npz"):
            name += "_MOCK"
        out[name] = npz
    return out


def parse_explicit(pairs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pair in pairs:
        if "=" not in pair:
            sys.exit(f"--features expects name=path, got {pair!r}")
        name, path = pair.split("=", 1)
        out[name] = Path(path)
    return out


def run_one(mf_root: Path, name: str, npz: Path, split: str, seed: int,
            epochs: int, hidden: int, extra: list[str]) -> Path | None:
    run_name = f"{name}__{split}__s{seed}"
    metrics_path = RUNS_DIR / f"{run_name}.json"
    cmd = [
        sys.executable, str(mf_root / "model.py"),
        "--features", str(npz.resolve()),
        "--split-level", split,
        "--epochs", str(epochs),
        "--hidden", str(hidden),
        "--seed", str(seed),
        "--save-metrics", str(metrics_path),
        "--run-name", run_name,
        *extra,
    ]
    print(f"\n=== {run_name} ===\n  {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  [warn] model.py failed for {run_name} (rc={proc.returncode})")
        return None
    return metrics_path if metrics_path.exists() else None


def mean_score(per_head: dict) -> tuple[float, int]:
    """Mean of higher-is-better head scores (excludes rmse). Returns (mean, n)."""
    vals = [e["score"] for e in per_head.values() if e.get("metric_kind") != "rmse"]
    return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def summarize(run_paths: list[Path]) -> None:
    runs = []
    for p in run_paths:
        try:
            runs.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    if not runs:
        print("no runs to summarize")
        return

    # method -> split -> list of (mean_score, per_head)
    agg: dict[str, dict[str, list]] = {}
    for r in runs:
        name = r.get("run_name", "")
        method = name.split("__")[0] if "__" in name else name
        split = r.get("split_level", "?")
        ms, _ = mean_score(r.get("per_head", {}))
        agg.setdefault(method, {}).setdefault(split, []).append((ms, r.get("per_head", {})))

    methods = sorted(agg)
    splits = ["species", "genus", "family"]
    present_splits = [s for s in splits if any(s in agg[m] for m in methods)]

    lines = ["# Evo2 vs domain-BPE — downstream trait comparison", ""]
    lines.append("Mean per-head test score (higher is better; rmse heads excluded), "
                 "averaged over seeds. Each genome is represented by frozen features "
                 "from the named method and scored on microbe-foundation's 21-head model.")
    lines.append("")
    lines.append("## Mean score by method x split")
    lines.append("")
    lines.append("| Method | " + " | ".join(present_splits) + " |")
    lines.append("|---|" + "---:|" * len(present_splits))
    for m in methods:
        cells = [f"`{m}`"]
        for s in present_splits:
            entries = agg[m].get(s, [])
            scores = [ms for ms, _ in entries if ms == ms]  # drop nan
            cells.append(f"{statistics.mean(scores):.4f}" if scores else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Head-level delta: domain-BPE vs Evo2 (real or _MOCK), species split if present.
    bpe = next((m for m in methods if m.startswith("domain_bpe")), None)
    evo = next((m for m in methods if m.startswith("evo2")), None)
    delta_split = "species" if "species" in present_splits else (
        present_splits[0] if present_splits else None)

    def head_means(method: str, split: str):
        acc: dict[str, list[float]] = {}
        kinds: dict[str, str] = {}
        for _ms, ph in agg[method].get(split, []):
            for h, e in ph.items():
                acc.setdefault(h, []).append(e["score"])
                kinds[h] = e.get("metric_kind", "?")
        return {h: sum(v) / len(v) for h, v in acc.items()}, kinds

    if bpe and evo and delta_split:
        lines.append(f"## Per-head delta (domain-BPE − Evo2), {delta_split} split, seed-averaged")
        lines.append("")
        lines.append("| Head | metric | Evo2 | domain-BPE | Δ (BPE−Evo2) |")
        lines.append("|---|---|---:|---:|---:|")
        evo_m, kinds = head_means(evo, delta_split)
        bpe_m, _ = head_means(bpe, delta_split)
        for h in sorted(set(evo_m) | set(bpe_m)):
            e = evo_m.get(h)
            b = bpe_m.get(h)
            d = (b - e) if (e is not None and b is not None) else None
            e_s = f"{e:.4f}" if e is not None else "—"
            b_s = f"{b:.4f}" if b is not None else "—"
            d_s = f"{d:+.4f}" if d is not None else "—"
            lines.append(f"| `{h}` | {kinds.get(h, '?')} | {e_s} | {b_s} | {d_s} |")
        lines.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    print("\n".join(lines[:12]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO / "data")
    p.add_argument("--mf-root", type=str, default=None)
    p.add_argument("--features", action="append", default=[],
                   help="name=path (repeatable). Default: auto-discover data/*features*.npz")
    p.add_argument("--splits", nargs="+", default=["species", "genus", "family"],
                   choices=["species", "genus", "family"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--skip-train", action="store_true",
                   help="don't run model.py; just re-summarize existing runs/")
    p.add_argument("--model-args", nargs=argparse.REMAINDER, default=[],
                   help="extra args passed through to model.py (after --model-args)")
    args = p.parse_args()

    mf_root = resolve_mf_root(args.mf_root)
    print(f"microbe-foundation: {mf_root}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    features = parse_explicit(args.features) if args.features else discover_features(args.data_dir)
    if not features:
        sys.exit(f"no feature npz found in {args.data_dir}. Run the extractors first.")
    print("feature sets:")
    for name, path in features.items():
        print(f"  {name:<16} {path}")

    run_paths: list[Path] = []
    if not args.skip_train:
        for name, npz in features.items():
            if not npz.exists():
                print(f"  [warn] missing {npz}, skipping {name}")
                continue
            for split in args.splits:
                for seed in args.seeds:
                    mp = run_one(mf_root, name, npz, split, seed,
                                 args.epochs, args.hidden, args.model_args)
                    if mp:
                        run_paths.append(mp)
    else:
        run_paths = sorted(RUNS_DIR.glob("*.json"))

    # microbe-foundation leaderboard over our runs. Note: leaderboard.py prints
    # args.out.relative_to(ROOT) at the end, which raises when --out lives outside
    # the microbe-foundation root (ours does). The file is written *before* that
    # line, so we capture output and treat a written file as success.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lb_out = RESULTS_DIR / "leaderboard.md"
    lb_cmd = [sys.executable, str(mf_root / "leaderboard.py"),
              "--runs-dir", str(RUNS_DIR), "--out", str(lb_out)]
    print(f"\n=== leaderboard ===\n  {' '.join(lb_cmd)}")
    proc = subprocess.run(lb_cmd, capture_output=True, text=True)
    if lb_out.exists():
        print(f"  wrote {lb_out}")
    else:
        print(f"  [warn] leaderboard.py did not produce {lb_out}")
        if proc.stderr:
            print(proc.stderr.strip())

    summarize(run_paths if run_paths else sorted(RUNS_DIR.glob("*.json")))


if __name__ == "__main__":
    main()
