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
    try:
        with open(json_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: file not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON in {json_path}: {e}")
        sys.exit(1)


def plot_true_vs_recovered(data: dict, out_dir: str):
    """Scatter plot: θ_true vs θ_rec for each parameter."""
    params = data["param_names"]
    results = data["results"]
    n_params = len(params)
    n_success = sum(1 for r in results if r.get("success", False))

    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i, pname in enumerate(params):
        ax = axes[i]
        true_vals = []
        rec_vals = []
        for r in results:
            if not r.get("success", False):
                continue
            theta_true = r.get("theta_true")
            theta_rec = r.get("theta_rec")
            if theta_true is None or theta_rec is None or i >= len(theta_true) or i >= len(theta_rec):
                continue
            tv = theta_true[i]
            rv = theta_rec[i]
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
        ax.set_xlabel("log-parameter value (true)", fontsize=6)
        ax.set_ylabel("log-parameter value (recovered)", fontsize=6)
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
        if not r.get("success", False):
            continue
        abs_err = r.get("abs_err_log")
        if abs_err is None:
            continue
        for i, pname in enumerate(params):
            if i >= len(abs_err):
                continue
            e = abs_err[i]
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
    bp = ax.boxplot(values, showmeans=True, meanprops={
        "marker": "o", "markerfacecolor": "red", "markersize": 3})
    ax.set_xticklabels(names, rotation=45, fontsize=6)
    ax.set_yscale("symlog")

    ax.set_ylabel("|θ_err| (log-space)")
    ax.set_title("Parameter Recovery Error by Parameter")
    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(out_dir, "error_boxplot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_error_vs_phys_magnitude(data: dict, out_dir: str):
    """Scatter: physical param value vs relative error (multi-panel)."""
    params = data["param_names"]
    results = data["results"]
    n_params = len(params)

    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i, pname in enumerate(params):
        ax = axes[i]
        phys_vals = []
        errs = []
        for r in results:
            if not r.get("success", False):
                continue
            ptp = r.get("param_true_phys")
            aep = r.get("abs_err_phys")
            if ptp is None or aep is None or i >= len(ptp) or i >= len(aep):
                continue
            pv = ptp[i]
            ae = aep[i]
            if pv is not None and ae is not None and pv > 1e-10:
                phys_vals.append(pv)
                errs.append(abs(ae / pv) * 100)  # relative %

        if phys_vals:
            ax.scatter(phys_vals, errs, alpha=0.6, s=15, edgecolors="k", linewidths=0.3)
        ax.set_xlabel(f"True {pname} value", fontsize=6)
        ax.set_ylabel("Rel. error (%)", fontsize=6)
        ax.set_title(pname, fontsize=8)
        ax.tick_params(axis="both", labelsize=5)
        ax.grid(alpha=0.3)

    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Relative error vs true physical parameter value", fontsize=10, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "relerr_vs_phys.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_gate_summary(data: dict, out_dir: str):
    """Bar chart of gate pass rates."""
    results = data["results"]
    n_total = len(results)

    ga_pass = sum(1 for r in results if r.get("gate_a", {}).get("passed", False))
    gb_pass = sum(1 for r in results if r.get("gate_b", {}).get("passed", False))
    gc_dominated = []
    for r in results:
        gc_dominated.append(r.get("gate_c", {}).get("n_prior_dominated", 0))
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

    succ = [r for r in results if r.get("success", False) and r.get("theta_rec") is not None]
    theta = [r["theta_rec"] for r in succ]
    costs = [r.get("cost", float("nan")) for r in succ]
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


def plot_cost_vs_clusters(data: dict, out_dir: str):
    """Scatter: cost vs n_clusters (gate A), annotated by patient ID."""
    results = data["results"]
    succ = [r for r in results if r.get("success", False)]
    costs = [r["cost"] for r in succ]
    n_cl = [r.get("gate_a", {}).get("n_clusters", 0) for r in succ]
    pids = [r.get("patient_id", -1) for r in succ]
    if not costs:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(n_cl, costs, s=50, c="steelblue", edgecolors="k", linewidths=0.5, zorder=3)
    for pid, nc, c in zip(pids, n_cl, costs):
        ax.annotate(str(pid), (nc, c), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.8)
    ax.set_xlabel("Number of CMA-ES clusters (gate A)")
    ax.set_ylabel("Final cost")
    ax.set_title("Cost vs. number of CMA-ES modes")
    ax.grid(alpha=0.3)
    ax.set_xticks(sorted(set(n_cl)))

    out_path = os.path.join(out_dir, "cost_vs_clusters.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_error_correlation_heatmap(data: dict, out_dir: str):
    """Heatmap of Pearson correlation between parameter errors."""
    params = data["param_names"]
    results = data["results"]
    errs = [r["abs_err_log"] for r in results if r.get("success", False)]
    errs = [e for e in errs if e is not None and all(v is not None for v in e)]
    if len(errs) < 3:
        print("  Too few valid error vectors for correlation heatmap")
        return
    X = np.array(errs)
    C = np.corrcoef(X.T)
    mask = np.triu(np.ones_like(C, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(params)))
    ax.set_yticks(range(len(params)))
    ax.set_xticklabels(params, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(params, fontsize=7)
    for i in range(len(params)):
        for j in range(len(params)):
            if not mask[i, j]:
                ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(C[i,j]) > 0.5 else "black")
    ax.set_title("Correlation of |θ_err| between parameters")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = os.path.join(out_dir, "error_correlation_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_pca_with_true(data: dict, out_dir: str):
    """PCA of recovered θ with true θ projected; lines connect each pair."""
    params = data["param_names"]
    results = data["results"]
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return

    succ = [r for r in results if r.get("success", False) and r.get("theta_rec") is not None and r.get("theta_true") is not None]
    theta_r = np.array([r["theta_rec"] for r in succ])
    theta_t = np.array([r["theta_true"] for r in succ])
    costs = np.array([r.get("cost", float("nan")) for r in succ])
    if len(theta_r) < 3:
        return

    all_theta = np.vstack([theta_r, theta_t])
    all_centered = all_theta - all_theta.mean(axis=0)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_centered)
    n = len(theta_r)
    rec_2d = coords[:n]
    true_2d = coords[n:]
    var_exp = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(rec_2d[:, 0], rec_2d[:, 1], c=costs, cmap="viridis",
                    s=40, edgecolors="k", linewidths=0.5, zorder=3, label="Recovered")
    ax.scatter(true_2d[:, 0], true_2d[:, 1], marker="x", c="red",
               s=40, linewidths=1, zorder=3, label="True")
    for i in range(n):
        ax.plot([true_2d[i, 0], rec_2d[i, 0]],
                [true_2d[i, 1], rec_2d[i, 1]],
                color="gray", linewidth=0.5, alpha=0.5, zorder=1)

    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    ax.set_title("PCA: recovered θ vs true θ")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Final cost")

    out_path = os.path.join(out_dir, "pca_vs_true.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_coefficient_variation(data: dict, out_dir: str):
    """Bar chart of coefficient of variation per parameter."""
    params = data["param_names"]
    results = data["results"]

    cv_vals = []
    for i, pname in enumerate(params):
        ratios = []
        for r in results:
            if not r.get("success", False):
                continue
            abs_err = r.get("abs_err_log")
            theta_true = r.get("theta_true")
            if abs_err is None or theta_true is None or i >= len(abs_err) or i >= len(theta_true):
                continue
            ae = abs_err[i]
            tv = abs(theta_true[i]) if theta_true[i] is not None else 0
            if ae is not None and tv > 1e-10:
                ratios.append(ae / tv)
        cv_vals.append(np.median(ratios) if ratios else 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.RdYlGn_r(1 - np.clip(np.array(cv_vals) / max(cv_vals), 0, 1))
    bars = ax.bar(range(len(params)), cv_vals, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(params, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Median |θ_err| / |θ_true|")
    ax.set_title("Coefficient of variation by parameter")
    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(out_dir, "coeff_variation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def print_detailed_gate_table(data: dict):
    """Print per-patient gate summary."""
    results = data["results"]
    headers = ["PID", "Cost", "nClust", "GateA", "GateB", "nDom", "LjungQ"]
    print(f"{headers[0]:>4} {headers[1]:>10} {headers[2]:>6} {headers[3]:>6} {headers[4]:>6} {headers[5]:>5} {headers[6]:>10}")
    print(f"{'-'*4} {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*10}")
    for r in results:
        pid = r.get("patient_id", -1)
        cost = r.get("cost", float("nan"))
        gate_a = r.get("gate_a", {})
        gate_b = r.get("gate_b", {})
        gate_c = r.get("gate_c", {})
        nc = gate_a.get("n_clusters", 0)
        ga = "PASS" if gate_a.get("passed", False) else "FAIL"
        gb = "PASS" if gate_b.get("passed", False) else "FAIL"
        nd = gate_c.get("n_prior_dominated", 0)
        lq = gate_b.get("lb_q", 0.0)
        print(f"{pid:>4} {cost:>10.4f} {nc:>6} {ga:>6} {gb:>6} {nd:>5} {lq:>10.1f}")
    print()


def print_summary_table(data: dict):
    """Print per-parameter recovery stats."""
    params = data["param_names"]
    results = data["results"]
    n_success = sum(1 for r in results if r.get("success", False))

    print(f"\n{'=' * 90}")
    print(f"{'Parameter':<12} {'n_rec':>5} {'Mean |err|':>12} {'Median |err|':>14} "
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


def plot_residual_timeseries(data: dict, out_dir: str, max_patients: int = 3):
    """CGM observed vs predicted + residuals + ACF for patients with lowest cost."""
    try:
        from statsmodels.graphics.tsaplots import plot_acf
    except ImportError:
        plot_acf = None

    results = [r for r in data["results"] if r["success"] and "cgm_residuals" in r]
    if not results:
        return
    results.sort(key=lambda r: r["cost"])
    n_plot = min(max_patients, len(results))

    for idx in range(n_plot):
        r = results[idx]
        n_obs = len(r["cgm_residuals"])
        time_hr = np.arange(n_obs) * 5 / 60

        cgm_obs = r.get("cgm_observed")
        cgm_pred = None
        if cgm_obs is not None:
            cgm_pred = np.array(cgm_obs) - np.array(r["cgm_residuals"])

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 0.8], hspace=0.25)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        ax2 = fig.add_subplot(gs[2], sharex=ax0)

        # --- Top: observed vs predicted ---
        if cgm_obs is not None and cgm_pred is not None:
            ax0.plot(time_hr, cgm_obs, color="black", linewidth=0.6, label="Observed (CGM)")
            ax0.plot(time_hr, cgm_pred, color="orange", linewidth=0.6, label="Predicted (model)")
            ax0.set_ylabel("Glucose [mg/dL]")
            ax0.legend(fontsize=7, loc="upper right")
            ax0.set_title(f"Patient {r['patient_id']}: CGM fit (cost={r['cost']:.4f})")
        else:
            ax0.text(0.5, 0.5, "CGM observed data not available",
                     transform=ax0.transAxes, ha="center", va="center")
            ax0.set_title(f"Patient {r['patient_id']}")
        ax0.grid(alpha=0.3)

        # --- Middle: residuals ---
        res_arr = np.array(r["cgm_residuals"])
        res_min = res_arr.min()
        res_max = res_arr.max()
        bound = max(abs(res_min), abs(res_max)) * 1.2

        ax1.axhline(0, color="gray", linewidth=0.3)
        ax1.fill_between(time_hr, 0, res_arr,
                         where=res_arr >= 0,
                         color="red", alpha=0.3, label="Overestimate")
        ax1.fill_between(time_hr, 0, res_arr,
                         where=res_arr < 0,
                         color="blue", alpha=0.3, label="Underestimate")
        ax1.plot(time_hr, res_arr, color="black", linewidth=0.5)
        ax1.set_ylabel("Residual [mg/dL]")
        ax1.set_ylim(-bound, bound)
        ax1.legend(fontsize=7)
        ax1.grid(alpha=0.3)

        # --- Bottom: ACF ---
        if plot_acf is not None:
            plot_acf(res_arr, ax=ax2, lags=min(40, n_obs // 5), alpha=0.05,
                     zero=False, linewidth=0.8)
            ax2.set_title("")
        else:
            n_lags = min(40, n_obs // 5)
            res_c = res_arr - res_arr.mean()
            var = res_c.var()
            acf = [1.0]
            if var > 1e-12:
                for lag in range(1, n_lags + 1):
                    acf.append(np.corrcoef(res_c[:-lag], res_c[lag:])[0, 1] if len(res_c) > lag else 0)
            ax2.bar(range(len(acf)), acf, width=0.6, color="steelblue", alpha=0.8)
            ax2.axhline(1.96 / np.sqrt(n_obs), color="red", linestyle="--", linewidth=0.5, alpha=0.6)
            ax2.axhline(-1.96 / np.sqrt(n_obs), color="red", linestyle="--", linewidth=0.5, alpha=0.6)
        ax2.set_xlabel("Time [hours]")
        ax2.set_ylabel("Autocorrelation")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"residuals_p{r['patient_id']}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


def plot_residual_acf(data: dict, out_dir: str):
    """Small-multiples ACF plots for all patients."""
    try:
        from statsmodels.graphics.tsaplots import plot_acf
    except ImportError:
        plot_acf = None

    results = [r for r in data["results"] if r["success"] and "cgm_residuals" in r]
    if not results:
        return

    n = len(results)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        res = np.array(r["cgm_residuals"])
        n_obs = len(res)
        if plot_acf is not None:
            plot_acf(res, ax=ax, lags=min(40, n_obs // 5), alpha=0.05,
                     zero=False, linewidth=0.8, color="steelblue")
            ax.set_title("")
        else:
            n_lags = min(40, n_obs // 5)
            res_c = res - res.mean()
            var = res_c.var()
            acf = [1.0]
            if var > 1e-12:
                for lag in range(1, n_lags + 1):
                    acf.append(np.corrcoef(res_c[:-lag], res_c[lag:])[0, 1] if len(res_c) > lag else 0)
            ax.bar(range(len(acf)), acf, width=0.6, color="steelblue", alpha=0.7)
            ax.axhline(1.96 / np.sqrt(n_obs), color="red", linestyle="--", linewidth=0.5)
            ax.axhline(-1.96 / np.sqrt(n_obs), color="red", linestyle="--", linewidth=0.5)
        ax.set_title(f"P{r['patient_id']} (cost={r['cost']:.2f})", fontsize=7)
        ax.tick_params(axis="both", labelsize=5)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Residual Autocorrelation Function (all patients)", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "acf_all.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_cmaes_landscape(data: dict, out_dir: str):
    """Small-multiples: PCA of CMA-ES restart points, colored by DBSCAN cluster."""
    results = [r for r in data["results"] if r.get("success", False) and "restarts" in r and len(r["restarts"]) >= 3]
    if not results:
        return

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return

    n = len(results)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        restarts = r["restarts"]
        X = np.array([rr["params"] for rr in restarts])
        costs = np.array([rr.get("cost", 0) for rr in restarts])
        gate_a = r.get("gate_a", {})
        labels = gate_a.get("labels", [-1] * len(restarts))

        Xc = X - X.mean(axis=0)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(Xc)
        var_exp = pca.explained_variance_ratio_

        n_clusters = gate_a.get("n_clusters", 0)
        cmap = plt.cm.tab10
        for pi in range(len(restarts)):
            lbl = labels[pi] if labels else -1
            color = cmap(lbl % 10) if lbl >= 0 else "gray"
            marker = "x" if lbl < 0 else "o"
            size = 50 if lbl >= 0 else 30
            ax.scatter(coords[pi, 0], coords[pi, 1],
                       c=[color], marker=marker, s=size,
                       edgecolors="k", linewidths=0.3,
                       alpha=0.8, zorder=3)

        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=5)
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=5)
        ax.tick_params(axis="both", labelsize=5)
        ax.set_title(f"P{r['patient_id']}: {n_clusters}cl, cost={r['cost']:.2f}", fontsize=7)
        ax.grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("CMA-ES landscape (PCA of restart points, colored by cluster)", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "cmaes_landscape_all.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


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
        plot_cost_vs_clusters(data, args.out)
        plot_error_correlation_heatmap(data, args.out)
        plot_pca_with_true(data, args.out)
        plot_coefficient_variation(data, args.out)
        plot_residual_timeseries(data, args.out)
        plot_residual_acf(data, args.out)
        plot_cmaes_landscape(data, args.out)

    print_detailed_gate_table(data)
    print_summary_table(data)


if __name__ == "__main__":
    main()
