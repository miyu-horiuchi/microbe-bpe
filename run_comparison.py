#!/usr/bin/env python3
"""Step 4 — run the downstream comparison and summarize.

For each feature set (single-nt, domain-BPE, Evo2[, Evo2-BPE]) x split
(species/genus/family) x seed, this drives microbe-foundation's model.py to train
the 21-head trait model on those frozen genome features, saving per-run metrics.
It then calls microbe-foundation's leaderboard.py and writes a summary whose
HEADLINE is the matched-capacity pair `single_nt` vs `domain_bpe` (only the
tokenizer differs); the Evo2 arms are reported as a larger single-nt reference.
The summary reports mean ± std over seeds and a per-trait-class breakdown so the
tokenization effect isn't washed out by averaging unrelated heads.

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


# Coarse trait taxonomy for the headline contrast. "machinery" traits encode
# gene content / functional machinery (where richer DNA tokens *might* help);
# "compositional" traits are closer to bulk sequence composition. We match on
# substrings of the head name so it's robust to the exact schema names.
MACHINERY_KEYS = (
    "pathogen", "medium", "cultivation", "carbon", "metabolite", "substrate",
    "amr", "resist", "antibiotic", "biosafety", "fame", "fatty", "enzyme",
)
COMPOSITIONAL_KEYS = (
    "gram", "shape", "motil", "spor", "oxygen", "aerob", "catalase", "oxidase",
    "temperature", "temp", "ph_", "halo", "salin", "pigment", "gc_",
)


def trait_class(head: str) -> str:
    h = head.lower()
    if any(k in h for k in MACHINERY_KEYS):
        return "machinery"
    if any(k in h for k in COMPOSITIONAL_KEYS):
        return "compositional"
    return "other"


def _wilcoxon(deltas: list[float]):
    """Two-sided Wilcoxon signed-rank p-value for deltas != 0 (None if unavailable)."""
    nz = [d for d in deltas if d != 0]
    if len(nz) < 6:
        return None
    try:
        from scipy.stats import wilcoxon  # optional dependency
    except Exception:
        return None
    try:
        return float(wilcoxon(nz).pvalue)
    except Exception:
        return None


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
    n_seeds = max((len(agg[m].get(s, [])) for m in methods for s in present_splits), default=0)

    def head_means(method: str, split: str):
        """Seed-averaged per-head score + metric kind for a method/split."""
        acc: dict[str, list[float]] = {}
        kinds: dict[str, str] = {}
        for _ms, ph in agg.get(method, {}).get(split, []):
            for h, e in ph.items():
                acc.setdefault(h, []).append(e["score"])
                kinds[h] = e.get("metric_kind", "?")
        return {h: sum(v) / len(v) for h, v in acc.items()}, kinds

    # The headline test is the matched-capacity pair: only the tokenizer differs.
    bpe = next((m for m in methods if m.startswith("domain_bpe")), None)
    nt = next((m for m in methods if m.startswith("single_nt")), None)
    evo = next((m for m in methods if m.startswith("evo2") and not m.startswith("evo2_bpe")), None)
    evo_bpe = next((m for m in methods if m.startswith("evo2_bpe")), None)

    lines = ["# Tokenization-trap downstream test — trait prediction", ""]
    lines.append(
        "Each genome is frozen into features by the named method, then scored on "
        "microbe-foundation's 21-head trait model. **Headline comparison:** "
        "`single_nt` vs `domain_bpe` — same TinyGPT, residue-matched training, "
        "only the tokenizer differs. Evo2 arms are a far-larger single-nucleotide "
        "*reference*, not a matched control.")
    lines.append("")
    lines.append(f"Seeds per cell: {n_seeds} (± is std over seeds). "
                 "Scores are mean per-head test score, higher is better, rmse heads excluded.")
    lines.append("")

    role = {}
    if nt: role[nt] = "headline control (single nucleotide)"
    if bpe: role[bpe] = "headline method (domain BPE)"
    if evo: role[evo] = "reference (Evo2, single-nt, ~billions params)"
    if evo_bpe: role[evo_bpe] = "reference (Evo2 embeddings pooled by BPE spans)"

    lines.append("## Mean score by method x split")
    lines.append("")
    lines.append("| Method | role | " + " | ".join(present_splits) + " |")
    lines.append("|---|---|" + "---:|" * len(present_splits))
    for m in methods:
        cells = [f"`{m}`", role.get(m, "—")]
        for s in present_splits:
            scores = [ms for ms, _ in agg[m].get(s, []) if ms == ms]  # drop nan
            if not scores:
                cells.append("—")
            elif len(scores) >= 2:
                cells.append(f"{statistics.mean(scores):.4f} ± {statistics.stdev(scores):.4f}")
            else:
                cells.append(f"{scores[0]:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Headline contrast by trait class: does richer tokenization help, and where?
    if bpe and nt:
        lines.append("## Headline contrast Δ(domain_bpe − single_nt) by trait class")
        lines.append("")
        lines.append("Mean Δ over the non-rmse heads in each class (± std across heads); "
                     "p = two-sided Wilcoxon signed-rank over those heads (needs scipy & ≥6 heads).")
        lines.append("")
        lines.append("| Split | trait class | n heads | mean Δ | p |")
        lines.append("|---|---|---:|---:|---:|")
        for s in present_splits:
            bpe_m, kinds = head_means(bpe, s)
            nt_m, _ = head_means(nt, s)
            by_class: dict[str, list[float]] = {}
            for h in set(bpe_m) & set(nt_m):
                if kinds.get(h) == "rmse":
                    continue
                by_class.setdefault(trait_class(h), []).append(bpe_m[h] - nt_m[h])
            for cls in ("machinery", "compositional", "other", "all"):
                d = (sum(by_class.values(), []) if cls == "all" else by_class.get(cls, []))
                if not d:
                    continue
                mu = statistics.mean(d)
                sd = statistics.stdev(d) if len(d) >= 2 else 0.0
                p = _wilcoxon(d)
                p_s = f"{p:.3f}" if p is not None else "—"
                lines.append(f"| {s} | {cls} | {len(d)} | {mu:+.4f} ± {sd:.4f} | {p_s} |")
        lines.append("")

    # Full per-head table for the headline pair (+ references), species split if present.
    delta_split = "species" if "species" in present_splits else (
        present_splits[0] if present_splits else None)
    ref_methods = [m for m in (nt, bpe, evo, evo_bpe) if m]
    if delta_split and nt and bpe:
        lines.append(f"## Per-head scores, {delta_split} split, seed-averaged")
        lines.append("")
        header = ["Head", "class", "metric"] + [f"`{m}`" for m in ref_methods] + ["Δ(bpe−nt)"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---|---|---|" + "---:|" * (len(ref_methods) + 1))
        means = {m: head_means(m, delta_split)[0] for m in ref_methods}
        _, kinds = head_means(bpe, delta_split)
        all_heads = sorted(set().union(*[set(means[m]) for m in ref_methods]))
        for h in all_heads:
            row = [f"`{h}`", trait_class(h), kinds.get(h, "?")]
            for m in ref_methods:
                v = means[m].get(h)
                row.append(f"{v:.4f}" if v is not None else "—")
            dv = (means[bpe].get(h), means[nt].get(h))
            row.append(f"{dv[0]-dv[1]:+.4f}" if None not in dv else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    print("\n".join(lines[:14]))


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
