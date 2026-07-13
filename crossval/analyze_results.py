#!/usr/bin/env python3
"""
Analyze cross-validation results from cross_validate.
Usage: python3 analyze_results.py [--json crossval_results.json]
"""

import json
import sys
import os
import math
import argparse
from collections import defaultdict

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("ERROR: install numpy + matplotlib: pip install numpy matplotlib")
    sys.exit(1)


def load_results(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def plot_true_vs_recovered(data: dict, out_dir: str):
    """Scatter plot: θ_true vs θ_rec for each parameter."""
    params = data["param_names"]
    results = data["results"]
    n_params = len(params)
    n_success = sum(1 for r in results if r["success"])

    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i, pname in enumerate(params):
        ax = axes[i]
        true_vals = []
        rec_vals = []
        for r in results:
            if not r["success"]:
                continue
            tv = r["theta_true"][i]
            rv = r["theta_rec"][i]
            if tv is not None and rv is not None:
                true_vals.append(tv)
                rec_vals.append(rv)

        if true_vals:
            ax.scatter(true_vals, rec_vals, alpha=0.6, s=20, edgecolors="k", linewidths=0.3)
            # unity line
            lo = min(min(true_vals), min(rec_vals))
            hi = max(max(true_vals), max(rec_vals))
            margin = (hi - lo) * 0.1 if hi > lo else 0.5
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    "r--", linewidth=0.8, alpha=0.5)
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)

        ax.set_title(pname, fontsize=8)
        ax.set_xlabel("θ true", fontsize=6)
        ax.set_ylabel("θ recovered", fontsize=6)
        ax.tick_params(axis="both", labelsize=5)
        ax.set_aspect("equal", adjustable="box")

    # hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"True vs Recovered θ (log-space) — {n_success} successful identifications",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "true_vs_recovered.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_error_boxplot(data: dict, out_dir: str):
    """Boxplot of |θ_err| per parameter."""
    params = data["param_names"]
    results = data["results"]

    err_by_param = defaultdict(list)
    for r in results:
        if not r["success"]:
            continue
        for i, pname in enumerate(params):
            e = r["abs_err_log"][i]
            if e is not None:
                err_by_param[pname].append(e)

    names = []
    values = []
    for pname in params:
        if pname in err_by_param and err_by_param[pname]:
            names.append(pname)
            values.append(err_by_param[pname])

    if not values:
        print("  No data for error boxplot")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    bp = ax.boxplot(values, labels=names, showmeans=True, meanprops={
        "marker": "o", "markerfacecolor": "red", "markersize": 3})

    ax.set_ylabel("|θ_err| (log-space)")
    ax.set_title("Parameter Recovery Error by Parameter")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(out_dir, "error_boxplot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_error_vs_phys_magnitude(data: dict, out_dir: str):
    """Scatter: physical param value vs relative error."""
    params = data["param_names"]
    results = data["results"]

    for i, pname in enumerate(params):
        phys_vals = []
        errs = []
        for r in results:
            if not r["success"]:
                continue
            pv = r["param_true_phys"][i]
            ae = r["abs_err_phys"][i]
            if pv is not None and ae is not None and pv > 1e-10:
                phys_vals.append(pv)
                errs.append(abs(ae / pv) * 100)  # relative %

        if len(phys_vals) < 3:
            continue

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(phys_vals, errs, alpha=0.6, s=15, edgecolors="k", linewidths=0.3)
        ax.set_xlabel(f"True {pname} value")
        ax.set_ylabel("Relative error (%)")
        ax.set_title(f"{pname}: relative error vs true value")
        ax.grid(alpha=0.3)

        out_path = os.path.join(out_dir, f"relerr_vs_phys_{pname}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_gate_summary(data: dict, out_dir: str):
    """Bar chart of gate pass rates."""
    results = data["results"]
    n_total = len(results)

    ga_pass = sum(1 for r in results if r["gate_a"]["passed"])
    gb_pass = sum(1 for r in results if r["gate_b"]["passed"])
    # gate C is not pass/fail, show average dominated
    gc_dominated = []
    for r in results:
        gc_dominated.append(r["gate_c"]["n_prior_dominated"])
    avg_gc = np.mean(gc_dominated) if gc_dominated else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Bar chart
    categories = ["Gate A\n(multiplicity)", "Gate B\n(structural adequacy)"]
    pass_rates = [ga_pass / n_total * 100, gb_pass / n_total * 100]
    bars = ax1.bar(categories, pass_rates, color=["#4CAF50", "#2196F3"], alpha=0.8)
    for bar, rate in zip(bars, pass_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{rate:.0f}% ({int(rate * n_total / 100)}/{n_total})",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Pass rate (%)")
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Gate Pass Rates")

    # Histogram of Gate C dominated params
    ax2.hist(gc_dominated, bins=range(0, max(gc_dominated) + 2),
             align="left", color="#FF9800", alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Number of prior-dominated params")
    ax2.set_ylabel("Patients")
    ax2.set_title(f"Gate C: prior-dominated params (avg {avg_gc:.1f})")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "gate_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_pca_tsne(data: dict, out_dir: str):
    """PCA and t-SNE of recovered θ matrix, colored by final cost."""
    params = data["param_names"]
    results = data["results"]

    theta = [r["theta_rec"] for r in results if r["success"]]
    costs = [r["cost"] for r in results if r["success"]]
    if len(theta) < 3:
        print("  Too few points for PCA/t-SNE")
        return

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError:
        print("  sklearn not available, skipping PCA/t-SNE")
        return

    X = np.array(theta)
    c = np.array(costs)
    Xc = X - X.mean(axis=0)

    n = max(2, min(5, len(X) - 1))
    pca = PCA(n_components=2)
    pca_c = pca.fit_transform(Xc)
    pca_var = pca.explained_variance_ratio_

    tsne = TSNE(n_components=2, random_state=42, perplexity=n)
    tsne_c = tsne.fit_transform(Xc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sc1 = ax1.scatter(pca_c[:, 0], pca_c[:, 1], c=c, cmap="viridis",
                      s=40, edgecolors="k", linewidths=0.3)
    ax1.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}%)")
    ax1.set_title("PCA of recovered θ (colored by cost)")
    ax1.grid(alpha=0.3)
    plt.colorbar(sc1, ax=ax1, label="Final cost")

    sc2 = ax2.scatter(tsne_c[:, 0], tsne_c[:, 1], c=c, cmap="viridis",
                      s=40, edgecolors="k", linewidths=0.3)
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.set_title("t-SNE of recovered θ (colored by cost)")
    ax2.grid(alpha=0.3)
    plt.colorbar(sc2, ax=ax2, label="Final cost")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "pca_tsne.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def print_summary_table(data: dict):
    """Print per-parameter recovery stats."""
    params = data["param_names"]
    results = data["results"]
    n_success = sum(1 for r in results if r["success"])

    print(f"\n{'=' * 90}")
    print(f"{'Parameter':<12} {'n_rec':>5} {'Mean |θ_err|':>12} {'Median |θ_err|':>14} "
          f"{'P25':>8} {'P75':>8} {'Mean rel%':>10}")
    print(f"{'=' * 90}")

    for i, pname in enumerate(params):
        errs = []
        phys_true = []
        phys_rec = []
        for r in results:
            if not r["success"]:
                continue
            e = r["abs_err_log"][i]
            if e is not None:
                errs.append(e)
                pt = r["param_true_phys"][i]
                pr = r["param_rec_phys"][i]
                if pt is not None and pr is not None and pt > 1e-10:
                    phys_true.append(pt)
                    phys_rec.append(pr)

        if not errs:
            continue

        mean_err = np.mean(errs)
        med_err = np.median(errs)
        p25 = np.percentile(errs, 25)
        p75 = np.percentile(errs, 75)

        # mean relative error in physical space
        rel_errs = [abs(pr - pt) / abs(pt) * 100 for pt, pr in zip(phys_true, phys_rec)]
        mean_rel = np.mean(rel_errs) if rel_errs else float("nan")

        print(f"{pname:<12} {len(errs):>5} {mean_err:>12.4f} {med_err:>14.4f} "
              f"{p25:>8.4f} {p75:>8.4f} {mean_rel:>9.2f}%")

    print(f"{'=' * 90}")
    print(f"\nTotal patients: {len(results)} | Successful identifications: {n_success}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze cross-validation results")
    parser.add_argument("--json", default="crossval_results.json",
                        help="Path to crossval JSON (default: crossval_results.json)")
    parser.add_argument("--out", default=".",
                        help="Output directory for plots (default: current dir)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plots, only print summary table")
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"ERROR: {args.json} not found. Run cross_validate first.")
        sys.exit(1)

    data = load_results(args.json)
    os.makedirs(args.out, exist_ok=True)

    if not args.no_plots:
        print("Generating plots...")
        plot_true_vs_recovered(data, args.out)
        plot_error_boxplot(data, args.out)
        plot_error_vs_phys_magnitude(data, args.out)
        plot_gate_summary(data, args.out)
        plot_pca_tsne(data, args.out)

    print_summary_table(data)


if __name__ == "__main__":
    main()
