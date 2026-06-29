#!/usr/bin/env python3
"""Generate publication figures for the Evo2-vs-BPE downstream result.

Reads runs/*.json (per-head trait scores) and data/*_features.meta.json
(intrinsic Zipf / nt-per-token), and writes paper/figs/*.{png,pdf}:

  fig_downstream_bars   mean per-head score by method x split (mean +/- std seeds)
  fig_zipf_ladder       intrinsic token-stat ladder (Zipf exponent + nt/token)
  fig_class_delta       Delta(domain_bpe - single_nt) per trait class x split
  fig_scatter           per-head: tiny domain-BPE vs 7B Evo2 (species split)

Only the consistent 364-genome arms are used:
  single_nt, kmer_4, domain_bpe_1024, evo2, evo2_bpe
(the stale 50-genome 'domain_bpe' run is ignored).
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
RUNS = REPO / "runs"
DATA = REPO / "data"
FIGS = REPO / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

METHODS = ["single_nt", "kmer_4", "domain_bpe_1024", "evo2", "evo2_bpe"]
LABELS = {
    "single_nt": "single-nt",
    "kmer_4": "k-mer (k=4)",
    "domain_bpe_1024": "domain-BPE",
    "evo2": "Evo2-7B (mean)",
    "evo2_bpe": "Evo2-7B (BPE)",
}
COLORS = {
    "single_nt": "#d1495b",       # red  (the trap)
    "kmer_4": "#edae49",          # amber (middle rung)
    "domain_bpe_1024": "#2e7d32", # green (the method)
    "evo2": "#3a6ea5",            # blue (reference)
    "evo2_bpe": "#8d6e9c",        # purple (reference)
}
SPLITS = ["species", "genus", "family"]

# Trait-class taxonomy (mirrors run_comparison.py)
MACHINERY = ("pathogen", "medium", "cultivation", "carbon", "metabolite",
             "substrate", "amr", "resist", "antibiotic", "biosafety", "fame",
             "fatty", "enzyme")
COMPOSITIONAL = ("gram", "shape", "motil", "spor", "oxygen", "aerob", "catalase",
                 "oxidase", "temperature", "temp", "ph_", "halo", "salin",
                 "pigment", "gc_")


def trait_class(head: str) -> str:
    h = head.lower()
    if any(k in h for k in MACHINERY):
        return "machinery"
    if any(k in h for k in COMPOSITIONAL):
        return "compositional"
    return "other"


def mean_score(per_head: dict) -> float:
    vals = [e["score"] for e in per_head.values() if e.get("metric_kind") != "rmse"]
    return sum(vals) / len(vals) if vals else float("nan")


def load_runs():
    """method -> split -> list of (mean_score, per_head) across seeds."""
    agg: dict = defaultdict(lambda: defaultdict(list))
    for p in sorted(RUNS.glob("*.json")):
        name = p.stem
        method = name.split("__")[0]
        if method not in METHODS:
            continue
        d = json.loads(p.read_text())
        split = d.get("split_level", "?")
        ph = d.get("per_head", {})
        agg[method][split].append((mean_score(ph), ph))
    return agg


def fig_downstream_bars(agg):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(SPLITS))
    w = 0.16
    for i, m in enumerate(METHODS):
        means, errs = [], []
        for s in SPLITS:
            scores = [ms for ms, _ in agg[m].get(s, []) if ms == ms]
            means.append(np.mean(scores) if scores else np.nan)
            errs.append(np.std(scores) if len(scores) > 1 else 0.0)
        ax.bar(x + (i - 2) * w, means, w, yerr=errs, capsize=3,
               label=LABELS[m], color=COLORS[m], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SPLITS])
    ax.set_ylabel("Mean per-head test score (higher = better)")
    ax.set_xlabel("Taxonomic holdout")
    ax.set_title("Downstream trait prediction across the tokenizer ladder (364 genomes)")
    ax.set_ylim(0.35, 0.62)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_downstream_bars")


def fig_zipf_ladder():
    methods = ["single_nt", "kmer_4", "domain_bpe_1024"]
    zipf, ntpt = [], []
    for m in methods:
        meta = json.loads((DATA / f"{m}_features.meta.json").read_text())
        zipf.append(meta.get("zipf_exponent", np.nan))
        ntpt.append(meta.get("nt_per_token", np.nan))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(methods))
    cols = [COLORS[m] for m in methods]
    bars = ax1.bar(x, zipf, 0.55, color=cols, edgecolor="black", linewidth=0.5,
                   label="Zipf exponent $\\alpha$")
    ax1.axhspan(1.0, 1.2, color="green", alpha=0.08)
    ax1.axhline(1.0, ls="--", lw=1, color="green")
    ax1.text(2.42, 1.02, "language-like\nband ($\\alpha\\approx1$)", fontsize=8,
             color="green", va="bottom", ha="right")
    for b, v in zip(bars, zipf):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                 ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[m] for m in methods])
    ax1.set_ylabel("Zipf exponent $\\alpha$  (token rank-frequency)")
    ax1.set_ylim(0, 1.35)
    ax1.set_title("Intrinsic: domain-BPE restores Zipfian token statistics on real genomic DNA")
    ax2 = ax1.twinx()
    ax2.plot(x, ntpt, "o--", color="black", lw=1.3, ms=7, label="nt / token")
    for xi, v in zip(x, ntpt):
        ax2.text(xi + 0.06, v, f"{v:.2f}", fontsize=8, va="center")
    ax2.set_ylabel("nucleotides per token")
    ax2.set_ylim(0, 5.2)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_zipf_ladder")


def fig_class_delta(agg):
    """Mean Delta(domain_bpe - single_nt) per trait class, per split."""
    classes = ["machinery", "compositional", "other"]
    data = {c: [] for c in classes}
    for s in SPLITS:
        bpe_runs = agg["domain_bpe_1024"].get(s, [])
        nt_runs = agg["single_nt"].get(s, [])
        # seed-average per head
        def head_avg(runs):
            acc = defaultdict(list)
            for _ms, ph in runs:
                for h, e in ph.items():
                    if e.get("metric_kind") != "rmse":
                        acc[h].append(e["score"])
            return {h: np.mean(v) for h, v in acc.items()}
        hb, hn = head_avg(bpe_runs), head_avg(nt_runs)
        per_class = defaultdict(list)
        for h in set(hb) & set(hn):
            per_class[trait_class(h)].append(hb[h] - hn[h])
        for c in classes:
            data[c].append(np.mean(per_class[c]) if per_class[c] else np.nan)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(SPLITS))
    w = 0.25
    ccol = {"machinery": "#2e7d32", "compositional": "#3a6ea5", "other": "#999999"}
    for i, c in enumerate(classes):
        ax.bar(x + (i - 1) * w, data[c], w, label=c, color=ccol[c],
               edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SPLITS])
    ax.set_ylabel("$\\Delta$ score  (domain-BPE $-$ single-nt)")
    ax.set_xlabel("Taxonomic holdout")
    ax.set_title("Where domain-BPE helps: per-head gap by trait class")
    ax.legend(fontsize=9, title="trait class")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_class_delta")


def fig_scatter(agg):
    """Per-head: tiny domain-BPE vs 7B Evo2 on the species split (seed-avg)."""
    def head_avg(runs):
        acc = defaultdict(list)
        kind = {}
        for _ms, ph in runs:
            for h, e in ph.items():
                if e.get("metric_kind") != "rmse":
                    acc[h].append(e["score"])
                    kind[h] = e.get("metric_kind")
        return {h: np.mean(v) for h, v in acc.items()}, kind
    hb, kind = head_avg(agg["domain_bpe_1024"].get("species", []))
    he, _ = head_avg(agg["evo2"].get("species", []))
    heads = sorted(set(hb) & set(he))
    xb = [he[h] for h in heads]      # Evo2 on x
    yb = [hb[h] for h in heads]      # domain-BPE on y
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    cls_col = {"machinery": "#2e7d32", "compositional": "#3a6ea5", "other": "#999999"}
    for h, xx, yy in zip(heads, xb, yb):
        ax.scatter(xx, yy, s=42, color=cls_col[trait_class(h)],
                   edgecolor="black", linewidth=0.4, zorder=3)
    lim = [0, 1.02]
    ax.plot(lim, lim, "--", color="gray", lw=1, zorder=1)
    ax.text(0.04, 0.92, "domain-BPE better\n(above diagonal)", fontsize=8, color="#2e7d32")
    ax.text(0.55, 0.06, "Evo2-7B better", fontsize=8, color="#3a6ea5")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Evo2-7B per-head score (single nucleotide, ~7B params)")
    ax.set_ylabel("domain-BPE per-head score (~3M params)")
    ax.set_title("Per-trait: a tiny domain-BPE model vs a 7B genome model\n(species split)")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color=cls_col[c], label=c,
                      markeredgecolor="black") for c in ["machinery", "compositional", "other"]]
    ax.legend(handles=handles, fontsize=8, title="trait class", loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_scatter")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote paper/figs/{name}.png / .pdf")


def main():
    agg = load_runs()
    fig_downstream_bars(agg)
    fig_zipf_ladder()
    fig_class_delta(agg)
    fig_scatter(agg)
    print("done")


if __name__ == "__main__":
    main()
