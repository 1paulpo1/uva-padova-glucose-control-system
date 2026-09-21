#!/usr/bin/env python3
"""
Analyze synthetic identifiability results from cross_validate (virtual patients).
Usage: python3 analyze_results.py [--json crossval_results.json]
"""

import json
import sys
import os
import argparse
import contextlib
import io
import logging

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    logging.error("ERROR: install numpy + matplotlib: pip install numpy matplotlib")
    sys.exit(1)


@contextlib.contextmanager
def _quiet_import():
    """Silence stderr (Python- and C-level) during an optional import.

    Broken third-party combos (e.g. packages built for NumPy 1.x running
    under NumPy 2.x) print full tracebacks from inside their own import
    handlers *before* our except fires. Suppressing fd 2 around the import
    keeps that noise out of the analysis log; our own warning below still
    reports the failure reason.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


def load_results(json_path: str) -> dict:
    try:
        with open(json_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"ERROR: file not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"ERROR: invalid JSON in {json_path}: {e}")
        sys.exit(1)


def get_successful(results):
    """Return list of successful identification results."""
    return [r for r in results if r.get("success", False)]


def precompute(results, n_params):
    """Build dict of centralized numpy arrays from a results list.

    Returns a dict with:
      - scalar arrays (N,): patient_id, cost, best_cma_cost, success,
        ga_pass, gb_pass, n_clusters, gc_dominated,
        total_evals, cmaes_evals, lm_evals, cost_improvement_pct
      - 2D arrays (N, P) or None: theta_true, theta_rec, abs_err_log,
        param_true_phys, param_rec_phys, abs_err_phys
    """
    n = len(results)
    p = n_params
    d = {}

    # Scalars
    d["patient_id"] = np.array([r.get("patient_id", -1) for r in results])
    d["cost"] = np.array([r.get("cost", float("nan")) for r in results], dtype=float)
    d["best_cma_cost"] = np.array([r.get("best_cma_cost", float("nan")) for r in results], dtype=float)
    d["success"] = np.array([r.get("success", False) for r in results], dtype=bool)
    d["ga_pass"] = np.array([r.get("gate_a", {}).get("passed", False) for r in results], dtype=bool)
    d["gb_pass"] = np.array([r.get("gate_b", {}).get("passed", False) for r in results], dtype=bool)
    d["n_clusters"] = np.array([r.get("gate_a", {}).get("n_clusters", 0) for r in results], dtype=int)
    d["gc_dominated"] = np.array([r.get("gate_c", {}).get("n_prior_dominated", 0) for r in results], dtype=int)
    # Per-parameter prior-dominated boolean matrix (N x P)
    pd_rows = []
    for r in results:
        pd_list = r.get("gate_c", {}).get("prior_dominated")
        if pd_list is None or len(pd_list) < p:
            pd_rows.append([False] * p)
        else:
            pd_rows.append(pd_list[:p])
    d["prior_dominated"] = np.array(pd_rows, dtype=bool)
    d["total_evals"] = np.array([r.get("total_evals", 0) for r in results], dtype=float)
    d["cmaes_evals"] = np.array([r.get("cmaes_evals", 0) for r in results], dtype=float)
    d["lm_evals"] = np.array([r.get("lm_evals", 0) for r in results], dtype=float)

    # Cost improvement (precomputed)
    bcc = d["best_cma_cost"]
    final = d["cost"]
    imp = np.full(n, float("nan"))
    mask = (bcc > 0) & (bcc < 1e10) & ~np.isnan(bcc) & ~np.isnan(final)
    imp[mask] = (bcc[mask] - final[mask]) / bcc[mask] * 100
    d["cost_improvement_pct"] = imp

    # Matrix fields (N x P or None)
    def build_matrix(key, null_ok=False):
        rows = []
        for r in results:
            v = r.get(key)
            if v is None or len(v) < p:
                rows.append(None)
            else:
                rows.append(v)
        if null_ok and all(r is None for r in rows):
            return None
        out = np.full((n, p), float("nan"))
        for i, row in enumerate(rows):
            if row is not None:
                for j in range(p):
                    if j < len(row) and row[j] is not None:
                        out[i, j] = row[j]
        return out

    d["theta_true"] = build_matrix("theta_true")
    d["theta_rec"] = build_matrix("theta_rec")
    d["abs_err_log"] = build_matrix("abs_err_log")
    d["param_true_phys"] = build_matrix("param_true_phys")
    d["param_rec_phys"] = build_matrix("param_rec_phys")
    d["abs_err_phys"] = build_matrix("abs_err_phys")

    return d


def filter_results(results, gate_a=None, gate_b=None, patient_range=None):
    """Filter results by success, gate passage, and patient ID range."""
    out = get_successful(results)
    if gate_a:
        out = [r for r in out if r.get("gate_a", {}).get("passed", False)]
    if gate_b:
        out = [r for r in out if r.get("gate_b", {}).get("passed", False)]
    if patient_range:
        lo, hi = patient_range
        out = [r for r in out if lo <= r.get("patient_id", -1) <= hi]
    return out


def plot_true_vs_recovered(data: dict, out_dir: str, min_half_span: float = 0.25):
    """Scatter plot: θ_true vs θ_rec for each parameter.

    Scale fix: axis limits are never tighter than ±``min_half_span`` around
    the data center (default 0.25 log-units = half of a typical prior std).
    Without this, a single-patient run auto-scales to a microscopic range and
    any point looks "on the line" — closeness to y=x is invisible. Each panel
    title additionally reports the recovery error so scale is explicit even
    at a glance.
    """
    params = data["param_names"]
    df = data["df"]
    n_params = len(params)
    n_success = int(df["success"].sum())

    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    theta_true = df["theta_true"]
    theta_rec = df["theta_rec"]
    valid = ~np.isnan(theta_true[:, 0]) & ~np.isnan(theta_rec[:, 0])

    all_err = []
    for i, pname in enumerate(params):
        ax = axes[i]
        tv = theta_true[valid, i]
        rv = theta_rec[valid, i]
        n_pts = len(tv)

        if n_pts > 0:
            err = np.abs(rv - tv)
            all_err.extend(err.tolist())
            med_err = float(np.median(err))
            label = f"|err|={err[0]:.3f}" if n_pts == 1 else f"med|err|={med_err:.3f}"
            ax.scatter(tv, rv, alpha=0.6, s=20, edgecolors="k", linewidths=0.3)
            lo = min(tv.min(), rv.min())
            hi = max(tv.max(), rv.max())
            center = (lo + hi) / 2.0
            # Never zoom tighter than ±min_half_span: keeps y=x closeness visible.
            half = max((hi - lo) / 2.0 * 1.1, min_half_span)
            ax.plot([center - half, center + half], [center - half, center + half],
                    "r--", linewidth=0.8, alpha=0.5)
            ax.set_xlim(center - half, center + half)
            ax.set_ylim(center - half, center + half)
        else:
            label = "no data"

        ax.set_title(f"{pname}  {label}", fontsize=8)
        ax.set_xlabel("log-parameter value (true)", fontsize=6)
        ax.set_ylabel("log-parameter value (recovered)", fontsize=6)
        ax.tick_params(axis="both", labelsize=5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    mean_err = float(np.mean(all_err)) if all_err else float("nan")
    fig.suptitle(f"True vs Recovered θ (log-space) — {n_success} successful identifications, "
                 f"mean |err|={mean_err:.3f} (axes ≥±{min_half_span} log-units)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "true_vs_recovered.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved {out_path}")


def plot_error_boxplot(data: dict, out_dir: str):
    """Boxplot of |θ_err| per parameter."""
    params = data["param_names"]
    err_mat = data["df"]["abs_err_log"]

    names = []
    values = []
    for i, pname in enumerate(params):
        col = err_mat[:, i]
        col = col[~np.isnan(col)]
        if len(col) > 0:
            names.append(pname)
            values.append(col)

    if not values:
        logging.warning("  No data for error boxplot")
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
    logging.info(f"  Saved {out_path}")


def plot_error_vs_phys_magnitude(data: dict, out_dir: str):
    """Scatter: physical param value vs relative error (multi-panel)."""
    params = data["param_names"]
    df = data["df"]
    n_params = len(params)
    ptp = df["param_true_phys"]
    aep = df["abs_err_phys"]
    valid = (ptp[:, 0] > 1e-10) & ~np.isnan(aep[:, 0])

    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i, pname in enumerate(params):
        ax = axes[i]
        pv = ptp[valid, i]
        ae = aep[valid, i]
        mask = (pv > 1e-10)
        pv = pv[mask]
        ae = ae[mask]
        if len(pv) > 0:
            ax.scatter(pv, abs(ae / pv) * 100, alpha=0.6, s=15, edgecolors="k", linewidths=0.3)
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
    logging.info(f"  Saved {out_path}")


def plot_gate_summary(data: dict, out_dir: str):
    """Bar chart of gate pass rates (uses raw/unfiltered data)."""
    d = data.get("inclusive_df") or data["raw_df"]
    n_total = len(d["ga_pass"])

    ga_pass = int(d["ga_pass"].sum())
    gb_pass = int(d["gb_pass"].sum())
    gc_dominated = d["gc_dominated"]
    avg_gc = float(gc_dominated.mean())

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
    xmax = min(max(gc_dominated) + 1, int(np.percentile(gc_dominated, 95)) + 2)
    bins = range(0, max(gc_dominated) + 2)
    n, _, patches = ax2.hist(gc_dominated, bins=bins,
                              align="left", color="#FF9800", alpha=0.8, edgecolor="white")
    ax2.set_xlim(-0.5, xmax - 0.5)
    ax2.set_xlabel("Number of prior-dominated params")
    ax2.set_ylabel("Patients")
    ax2.set_title(f"Gate C: prior-dominated params (avg {avg_gc:.1f})")
    ax2.grid(axis="y", alpha=0.3)

    # Cumulative fraction
    ax3 = ax2.twinx()
    cumsum = np.cumsum(n) / sum(n) * 100
    ax3.plot(range(len(cumsum)), cumsum, color="navy", marker="o", markersize=3,
             linewidth=0.8, label="Cumulative %")
    ax3.set_ylabel("Cumulative %", color="navy")
    ax3.tick_params(axis="y", colors="navy")
    ax3.set_ylim(0, 105)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "gate_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved {out_path}")


def plot_pca_tsne(data: dict, out_dir: str):
    """PCA and t-SNE of recovered θ matrix, colored by final cost."""
    df = data["df"]
    valid = ~np.isnan(df["theta_rec"][:, 0])
    theta = df["theta_rec"][valid]
    costs = df["cost"][valid]
    if len(theta) < 3:
        logging.warning("  Too few points for PCA/t-SNE")
        return

    try:
        with _quiet_import():
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
    except Exception as e:
        # Broad catch: broken envs (e.g. sklearn built for NumPy 1.x running
        # under NumPy 2.x) raise AttributeError/ImportError at import time.
        logging.warning(f"  sklearn import failed ({e}), skipping PCA/t-SNE")
        return

    Xc = theta - theta.mean(axis=0)

    n = max(2, min(5, len(theta) - 1))
    pca = PCA(n_components=2)
    pca_c = pca.fit_transform(Xc)
    pca_var = pca.explained_variance_ratio_

    tsne = TSNE(n_components=2, random_state=42, perplexity=n)
    tsne_c = tsne.fit_transform(Xc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sc1 = ax1.scatter(pca_c[:, 0], pca_c[:, 1], c=costs, cmap="viridis",
                      s=40, edgecolors="k", linewidths=0.3)
    ax1.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}%)")
    ax1.set_title("PCA of recovered θ (colored by cost)")
    ax1.grid(alpha=0.3)
    plt.colorbar(sc1, ax=ax1, label="Final cost")

    sc2 = ax2.scatter(tsne_c[:, 0], tsne_c[:, 1], c=costs, cmap="viridis",
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
    logging.info(f"  Saved {out_path}")


def plot_cost_vs_clusters(data: dict, out_dir: str):
    """Scatter: cost vs n_clusters (gate A), annotated by patient ID."""
    df = data["df"]
    costs = df["cost"]
    n_cl = df["n_clusters"]
    pids = df["patient_id"]
    if len(costs) == 0:
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
    logging.info(f"  Saved {out_path}")


def plot_error_correlation_heatmap(data: dict, out_dir: str):
    """Heatmap of Pearson correlation between parameter errors."""
    params = data["param_names"]
    err_mat = data["df"]["abs_err_log"]
    valid = ~np.isnan(err_mat).any(axis=1)
    err_mat = err_mat[valid]
    if len(err_mat) < 3:
        logging.warning("  Too few valid error vectors for correlation heatmap")
        return
    C = np.corrcoef(err_mat.T)
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
    logging.info(f"  Saved {out_path}")


def plot_pca_with_true(data: dict, out_dir: str):
    """PCA of recovered θ with true θ projected; lines connect each pair."""
    df = data["df"]
    valid = ~np.isnan(df["theta_rec"][:, 0]) & ~np.isnan(df["theta_true"][:, 0])
    if int(valid.sum()) < 3:
        return

    try:
        with _quiet_import():
            from sklearn.decomposition import PCA
    except Exception as e:
        # Broad catch: broken envs (e.g. sklearn built for NumPy 1.x running
        # under NumPy 2.x) raise AttributeError/ImportError at import time.
        logging.warning(f"  sklearn import failed ({e}), skipping PCA vs true")
        return

    theta_r = df["theta_rec"][valid]
    theta_t = df["theta_true"][valid]
    costs = df["cost"][valid]

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
    logging.info(f"  Saved {out_path}")


def plot_coefficient_variation(data: dict, out_dir: str):
    """Bar chart of coefficient of variation per parameter."""
    params = data["param_names"]
    df = data["df"]
    abs_err = df["abs_err_log"]
    theta_t = df["theta_true"]
    tv_abs = np.abs(theta_t)
    valid = ~np.isnan(abs_err) & ~np.isnan(theta_t) & (tv_abs > 1e-10)
    ratios = np.where(valid, abs_err / tv_abs, np.nan)
    cv_vals = [np.nanmedian(ratios[:, i]) if np.any(valid[:, i]) else 0 for i in range(len(params))]

    fig, ax = plt.subplots(figsize=(8, 4))
    cap = np.percentile(cv_vals, 95) if len(cv_vals) > 1 else max(cv_vals)
    cap = max(cap, 1e-6)
    colors = plt.cm.RdYlGn_r(1 - np.clip(np.array(cv_vals) / cap, 0, 1))
    bars = ax.bar(range(len(params)), cv_vals, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(params, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Median |θ_err| / |θ_true|")
    ax.set_title("Coefficient of variation by parameter")
    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(out_dir, "coeff_variation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved {out_path}")


def print_detailed_gate_table(data: dict):
    """Print per-patient gate summary (uses raw/unfiltered data)."""
    d = data.get("inclusive_df") or data["raw_df"]
    headers = ["PID", "Cost", "nClust", "GateA", "GateB", "nDom", "LjungQ"]
    logging.info(f"{headers[0]:>4} {headers[1]:>10} {headers[2]:>6} {headers[3]:>6} {headers[4]:>6} {headers[5]:>5} {headers[6]:>10}")
    logging.info(f"{'-'*4} {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*10}")
    # Gate B lb_q is not in precompute; fallback to raw data
    raw = data.get("results_inclusive") or data["results_raw"]
    for i, r in enumerate(raw):
        pid = d["patient_id"][i]
        cost = d["cost"][i]
        nc = d["n_clusters"][i]
        ga = d["ga_pass"][i]
        gb = d["gb_pass"][i]
        nd = d["gc_dominated"][i]
        lq = r.get("gate_b", {}).get("lb_q", 0.0)
        logging.info(f"{int(pid):>4} {cost:>10.4f} {nc:>6} {'PASS' if ga else 'FAIL':>6} {'PASS' if gb else 'FAIL':>6} {nd:>5} {lq:>10.1f}")
    logging.info("")


def print_prior_dominated_ranking(data: dict):
    """Print per-parameter ranking by prior-dominated frequency (from raw/unfiltered data)."""
    d = data.get("inclusive_df") or data["raw_df"]
    params = data["param_names"]
    pd_mat = d.get("prior_dominated")
    if pd_mat is None or pd_mat.size == 0:
        return
    n = pd_mat.shape[0]

    pct = [float(pd_mat[:, i].sum()) / n * 100 for i in range(len(params))]
    pairs = sorted(zip(params, pct), key=lambda x: -x[1])

    logging.info(f"\n{'=' * 50}")
    logging.info(f"{'Parameter':<12} {'Dom%':>7}   {'Status'}")
    logging.info(f"{'=' * 50}")
    for pname, dom_pct in pairs:
        status = "prior-dominated" if dom_pct >= 80 else "data-informed"
        logging.info(f"{pname:<12} {dom_pct:>6.1f}%   {status}")
    logging.info(f"{'=' * 50}\n")


def print_summary_table(data: dict):
    """Print per-parameter recovery stats (uses filtered/analyzed results)."""
    params = data["param_names"]
    df = data["df"]
    n_success = len(df["cost"])
    abs_err = df["abs_err_log"]
    ptp = df["param_true_phys"]
    prp = df["param_rec_phys"]
    pd_mat = df.get("prior_dominated")

    logging.info(f"\n{'=' * 97}")
    logging.info(f"{'Parameter':<12} {'n_rec':>5} {'Dom%':>6} {'Mean |err|':>12} {'Median |err|':>14} "
          f"{'P25':>8} {'P75':>8} {'Mean rel%':>10}")
    logging.info(f"{'=' * 97}")

    for i, pname in enumerate(params):
        errs = abs_err[:, i]
        errs = errs[~np.isnan(errs)]
        if len(errs) == 0:
            continue

        mean_err = np.mean(errs)
        med_err = np.median(errs)
        p25_val = np.percentile(errs, 25)
        p75_val = np.percentile(errs, 75)

        # mean relative error in physical space
        pt = ptp[:, i]
        pr = prp[:, i]
        mask = (pt > 1e-10) & ~np.isnan(pr) & ~np.isnan(pt)
        if mask.any():
            mean_rel = float(np.mean(np.abs(pr[mask] - pt[mask]) / np.abs(pt[mask]) * 100))
        else:
            mean_rel = float("nan")

        # prior-dominated percentage for this parameter
        if pd_mat is not None and pd_mat.shape[0] > 0:
            dom_pct = float(pd_mat[:, i].sum()) / pd_mat.shape[0] * 100
        else:
            dom_pct = float("nan")

        logging.info(f"{pname:<12} {len(errs):>5} {dom_pct:>5.1f}% {mean_err:>12.4f} {med_err:>14.4f} "
              f"{p25_val:>8.4f} {p75_val:>8.4f} {mean_rel:>9.2f}%")

    logging.info(f"{'=' * 97}")
    logging.info(f"\nTotal patients: {len(df['cost'])} | Successful identifications: {n_success}\n")


def plot_residual_timeseries(data: dict, out_dir: str, max_patients: int = 3):
    """CGM observed vs predicted + residuals + ACF for patients with lowest cost."""
    try:
        with _quiet_import():
            from statsmodels.graphics.tsaplots import plot_acf
    except Exception as e:
        # Broad catch: version-mismatched statsmodels raises TypeError, not
        # ImportError. Manual ACF fallback below covers the missing plotting.
        logging.warning(f"  statsmodels import failed ({e}), using manual ACF")
        plot_acf = None

    results = [r for r in data["results"] if "cgm_residuals" in r]
    if not results:
        return
    results.sort(key=lambda r: r["cost"])
    n_plot = min(max_patients, len(results))

    for idx in range(n_plot):
        r = results[idx]
        n_obs = len(r["cgm_residuals"])
        time_hr = np.arange(n_obs) * 5 / 60

        cgm_raw = r.get("input", {}).get("cgm_value") or r.get("true_input", {}).get("cgm_value")
        cgm_pred = None
        if cgm_raw is not None:
            cgm_obs = np.array(cgm_raw)
            cgm_pred = cgm_obs - np.array(r["cgm_residuals"])

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 0.8], hspace=0.25)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        # NB: no sharex here — the bottom panel is lag-domain (0..n_lags),
        # not time-domain. Sharing x stretched the time panels into empty
        # space and mislabeled the axis.
        ax2 = fig.add_subplot(gs[2])

        # --- Top: observed vs predicted ---
        if cgm_pred is not None:
            ax0.plot(time_hr, cgm_obs, color="black", linewidth=0.6, label="Observed (CGM)")
            ax0.plot(time_hr, cgm_pred, color="orange", linewidth=0.6, label="Predicted (model)")
            ax0.set_ylabel("Glucose [mg/dL]")
            ax0.legend(fontsize=7, loc="upper right")
            ax0.set_title(f"Patient {r['patient_id']}: CGM fit (cost={r['cost']:.4f})")
        else:
            ax0.text(0.5, 0.5, "CGM data not available",
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
        ax2.set_xlabel("Lag (CGM steps)")
        ax2.set_ylabel("Autocorrelation")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"residuals_p{r['patient_id']}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"  Saved {out_path}")


def plot_residual_acf(data: dict, out_dir: str):
    """Small-multiples ACF plots for all patients."""
    try:
        with _quiet_import():
            from statsmodels.graphics.tsaplots import plot_acf
    except Exception as e:
        # Broad catch: version-mismatched statsmodels raises TypeError, not
        # ImportError. Manual ACF fallback below covers the missing plotting.
        logging.warning(f"  statsmodels import failed ({e}), using manual ACF")
        plot_acf = None

    results = [r for r in data["results"] if "cgm_residuals" in r]
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
    logging.info(f"  Saved {out_path}")


def plot_cmaes_landscape(data: dict, out_dir: str):
    """Small-multiples: PCA of CMA-ES restart points, colored by DBSCAN cluster."""
    results = [r for r in data["results"] if "restarts" in r and len(r["restarts"]) >= 3]
    if not results:
        return

    try:
        with _quiet_import():
            from sklearn.decomposition import PCA
    except Exception as e:
        logging.warning(f"  sklearn import failed ({e}), skipping CMA-ES landscape")
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
    logging.info(f"  Saved {out_path}")


def plot_evals_histogram(data: dict, out_dir: str):
    """Stacked histogram of CMA-ES vs LM evaluations per patient."""
    d = data.get("inclusive_df") or data["df"]
    n = len(d["cmaes_evals"])
    has_lm = n > 0 and (d["cmaes_evals"] != 0).any()

    fig, ax = plt.subplots(figsize=(8, 4))
    if has_lm:
        cmaes = d["cmaes_evals"]
        lm = d["lm_evals"]
        total = cmaes + lm
        idx = np.argsort(total)
        ax.bar(range(n), cmaes[idx], label="CMA-ES evals", color="#2196F3", alpha=0.8)
        ax.bar(range(n), lm[idx], bottom=cmaes[idx], label="LM evals", color="#FF9800", alpha=0.8)
        ax.set_ylabel("Evaluations")
        ax.legend(fontsize=7)
        ax.set_title(f"CMA-ES vs LM evaluations per patient (n={n})")
    else:
        ax.hist(d["total_evals"], bins=30, color="steelblue", alpha=0.8, edgecolor="white")
        ax.set_xlabel("Total evaluations")
        ax.set_ylabel("Patients")
        ax.set_title(f"Total evaluations histogram (n={n})")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "evals_histogram.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved {out_path}")


def plot_cost_improvement(data: dict, out_dir: str):
    """Boxplot of cost improvement after LM refinement."""
    df = data["df"]
    improvements = df["cost_improvement_pct"]
    improvements = improvements[~np.isnan(improvements)]
    if len(improvements) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(improvements, vert=True, showmeans=True, meanprops={
        "marker": "o", "markerfacecolor": "red", "markersize": 4})
    ax.set_ylabel("Cost improvement (%)")
    ax.set_title(f"LM refinement: relative cost improvement (n={len(improvements)})")
    improved = sum(1 for v in improvements if v > 0.01)
    ax.text(0.5, 0.95, f"LM improved: {improved}/{len(improvements)} ({improved/len(improvements)*100:.0f}%)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.set_xticks([])

    out_path = os.path.join(out_dir, "cost_improvement.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved {out_path}")


def plot_evals_vs_accuracy(data: dict, out_dir: str):
    """Scatter: total_evals vs mean abs error, colored by Gate A."""
    df = data["df"]
    err_mat = df["abs_err_log"]
    valid = ~np.isnan(err_mat).all(axis=1)
    x = df["total_evals"][valid]
    y = np.nanmean(err_mat[valid], axis=1)
    gate_a_pass = df["ga_pass"][valid]

    if len(x) < 3:
        return

    colors = ["#4CAF50" if p else "#F44336" for p in gate_a_pass]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, c=colors, s=30, alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Total evaluations")
    ax.set_ylabel("Mean |θ_err| (log-space)")
    ax.set_title("Computational cost vs estimation accuracy")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4CAF50", label="Gate A PASS"),
                       Patch(facecolor="#F44336", label="Gate A FAIL")]
    ax.legend(handles=legend_elements, fontsize=7)
    ax.grid(alpha=0.3)

    out_path = os.path.join(out_dir, "evals_vs_accuracy.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved {out_path}")


def plot_error_by_gate(data: dict, out_dir: str):
    """Grouped bar: mean |err| per parameter split by gate passage."""
    params = data["param_names"]
    df = data["df"]
    err_mat = df["abs_err_log"]
    n_params = len(params)

    def nanmean_or_nan(arr):
        if len(arr) == 0:
            return np.full(n_params, float("nan"))
        return np.nanmean(arr, axis=0)

    ga = df["ga_pass"]
    gb = df["gb_pass"]
    gc = df["gc_dominated"]

    e_ga_p = nanmean_or_nan(err_mat[ga])
    e_ga_f = nanmean_or_nan(err_mat[~ga])
    e_gb_p = nanmean_or_nan(err_mat[gb])
    e_gb_f = nanmean_or_nan(err_mat[~gb])

    low  = gc <= 2
    med  = (gc >= 3) & (gc <= 5)
    high = gc >= 6

    e_gc_l = nanmean_or_nan(err_mat[low])
    e_gc_m = nanmean_or_nan(err_mat[med])
    e_gc_h = nanmean_or_nan(err_mat[high])

    x = np.arange(n_params)
    width = 0.22

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    ax1.bar(x - width/2, e_ga_p, width, label=f"PASS (n={ga.sum()})", color="#4CAF50", alpha=0.8)
    if (~ga).sum() > 0:
        ax1.bar(x + width/2, e_ga_f, width, label=f"FAIL (n={(~ga).sum()})", color="#F44336", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(params, rotation=45, fontsize=7)
    ax1.set_ylabel("Mean |θ_err|")
    ax1.set_title("Gate A (multiplicity)")
    ax1.legend(fontsize=6)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x - width/2, e_gb_p, width, label=f"PASS (n={gb.sum()})", color="#4CAF50", alpha=0.8)
    if (~gb).sum() > 0:
        ax2.bar(x + width/2, e_gb_f, width, label=f"FAIL (n={(~gb).sum()})", color="#F44336", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(params, rotation=45, fontsize=7)
    ax2.set_ylabel("Mean |θ_err|")
    ax2.set_title("Gate B (structural adequacy)")
    ax2.legend(fontsize=6)
    ax2.grid(axis="y", alpha=0.3)

    ax3.bar(x - width, e_gc_l, width, label=f"0-2 dom (n={low.sum()})", color="#4CAF50", alpha=0.8)
    ax3.bar(x,        e_gc_m, width, label=f"3-5 dom (n={med.sum()})", color="#FF9800", alpha=0.8)
    ax3.bar(x + width, e_gc_h, width, label=f"6+ dom (n={high.sum()})", color="#F44336", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(params, rotation=45, fontsize=7)
    ax3.set_ylabel("Mean |θ_err|")
    ax3.set_title("Gate C (prior-dominated params)")
    ax3.legend(fontsize=6)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "error_by_gate.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved {out_path}")


def plot_prior_dominated_bar(data: dict, out_dir: str):
    """Horizontal bar chart of prior-dominated frequency per parameter."""
    d = data.get("inclusive_df") or data["raw_df"]
    params = data["param_names"]
    pd_mat = d.get("prior_dominated")
    if pd_mat is None or pd_mat.size == 0:
        return
    n = pd_mat.shape[0]
    pct = [float(pd_mat[:, i].sum()) / n * 100 for i in range(len(params))]
    pairs = sorted(zip(params, pct), key=lambda x: -x[1])
    sorted_params = [p[0] for p in pairs]
    sorted_pct = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [plt.cm.RdYlGn_r(p / 100) for p in sorted_pct]
    bars = ax.barh(range(len(sorted_params)), sorted_pct, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(sorted_params)))
    ax.set_yticklabels(sorted_params, fontsize=9)
    ax.set_xlabel("Patients where parameter is prior-dominated (%)")
    ax.set_title(f"Prior-dominated parameter ranking (n={n})")
    ax.set_xlim(0, 105)
    for bar, pval in zip(bars, sorted_pct):
        ax.text(min(pval + 2, 95), bar.get_y() + bar.get_height() / 2,
                f"{pval:.0f}%", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "prior_dominated_bar.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved {out_path}")


PLOT_REGISTRY = {
    "true_vs_recovered": plot_true_vs_recovered,
    "error_boxplot": plot_error_boxplot,
    "error_vs_phys_magnitude": plot_error_vs_phys_magnitude,
    "gate_summary": plot_gate_summary,
    "pca_tsne": plot_pca_tsne,
    "cost_vs_clusters": plot_cost_vs_clusters,
    "error_correlation_heatmap": plot_error_correlation_heatmap,
    "pca_vs_true": plot_pca_with_true,
    "coeff_variation": plot_coefficient_variation,
    "residual_timeseries": plot_residual_timeseries,
    "acf_all": plot_residual_acf,
    "cmaes_landscape": plot_cmaes_landscape,
    "evals_histogram": plot_evals_histogram,
    "cost_improvement": plot_cost_improvement,
    "evals_vs_accuracy": plot_evals_vs_accuracy,
    "error_by_gate": plot_error_by_gate,
    "prior_dominated_bar": plot_prior_dominated_bar,
}


def main():
    parser = argparse.ArgumentParser(description="Analyze cross-validation results")
    parser.add_argument("--json", default="synthetic_results.json",
                        help="Path to synthetic study JSON (default: synthetic_results.json, alias crossval_results.json)")
    parser.add_argument("--out", default=".",
                        help="Output directory for plots (default: current dir)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plots, only print summary table")
    parser.add_argument("--plot", default="",
                        help="Comma-separated plot names to generate (default: all). "
                             f"Available: {', '.join(PLOT_REGISTRY)}")
    parser.add_argument("--gate-a", action="store_true",
                        help="Only include patients passing Gate A")
    parser.add_argument("--gate-b", action="store_true",
                        help="Only include patients passing Gate B")
    parser.add_argument("--patient-start", type=int, default=None,
                        help="Minimum patient ID (inclusive)")
    parser.add_argument("--patient-end", type=int, default=None,
                        help="Maximum patient ID (inclusive)")
    parser.add_argument("--include-failures", action="store_true",
                        help="Include failed identifications in gate_summary and evals_histogram")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s")

    if not os.path.exists(args.json):
        logging.error(f"ERROR: {args.json} not found. Run cross_validate first.")
        sys.exit(1)

    data = load_results(args.json)
    os.makedirs(args.out, exist_ok=True)

    # Preserve original unfiltered list for gate summary etc.
    data["results_raw"] = data["results"]

    # Apply filters
    patient_range = None
    if args.patient_start is not None or args.patient_end is not None:
        patient_range = (args.patient_start or 0, args.patient_end or 999999)

    # Recovery analysis: success + gate/range filters
    filtered = filter_results(data["results"], gate_a=args.gate_a,
                              gate_b=args.gate_b, patient_range=patient_range)
    if patient_range:
        logging.info(f"Patient range: {patient_range}")
    if args.gate_a or args.gate_b:
        logging.info(f"Filtered: {len(get_successful(data['results_raw']))} successful "
                     f"-> {len(filtered)} after gate/range filters")
    data["results"] = filtered

    n_params = len(data["param_names"])
    # Precompute: raw (unfiltered), filtered (for recovery plots), inclusive (if applicable)
    data["raw_df"] = precompute(data["results_raw"], n_params)
    data["df"] = precompute(filtered, n_params)

    # Inclusive set: all patients passing gate/range (including failures)
    if args.include_failures:
        inclu = [r for r in data["results_raw"] if r.get("success", False) or True]
        if args.gate_a:
            inclu = [r for r in inclu if r.get("gate_a", {}).get("passed", False)]
        if args.gate_b:
            inclu = [r for r in inclu if r.get("gate_b", {}).get("passed", False)]
        if patient_range:
            lo, hi = patient_range
            inclu = [r for r in inclu if lo <= r.get("patient_id", -1) <= hi]
        data["results_inclusive"] = inclu
        data["inclusive_df"] = precompute(inclu, n_params)
        logging.info(f"Inclusive (incl. failures): {len(inclu)} patients")

    if not args.no_plots:
        if args.plot:
            names = [n.strip() for n in args.plot.split(",")]
            unknown = [n for n in names if n not in PLOT_REGISTRY]
            if unknown:
                logging.error(f"Unknown plot(s): {unknown}. Available: {list(PLOT_REGISTRY)}")
                sys.exit(1)
        else:
            names = list(PLOT_REGISTRY)

        logging.info("Generating plots...")
        for name in names:
            try:
                PLOT_REGISTRY[name](data, args.out)
            except Exception as e:
                # One broken plot (e.g. env mismatch) must not kill the rest.
                logging.warning(f"  Plot '{name}' failed ({e}), skipping")

    print_detailed_gate_table(data)
    print_summary_table(data)
    print_prior_dominated_ranking(data)


if __name__ == "__main__":
    main()
