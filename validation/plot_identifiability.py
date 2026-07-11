#!/usr/bin/env python3
"""
Plot identifiability experiment results.
Reads CSV from stdin or file, produces box-plots.

Usage:
  cat exp_results.csv | python3 scripts/plot_identifiability.py
  python3 scripts/plot_identifiability.py < exp_results.csv
  python3 scripts/plot_identifiability.py /path/to/exp_results.csv
"""
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Parse CSV
rows = []
src = sys.argv[1] if len(sys.argv) > 1 else sys.stdin
reader = csv.DictReader(src)
for r in reader:
    r["rel_err_kp2"] = float(r["rel_err_kp2"])
    r["rel_err_Vm0"] = float(r["rel_err_Vm0"])
    r["rel_err_p2U"] = float(r["rel_err_p2U"])
    r["corr_Vm0_p2U"] = float(r["corr_Vm0_p2U"])
    r["final_cost"] = float(r["final_cost"])
    rows.append(r)

configs = sorted(set(r["config"] for r in rows), key=lambda c: c)
params = ["kp2", "Vm0", "p2U"]

# Color palette
colors = {"240_1meal": "#e74c3c", "720_3meals": "#3498db", "720_3meals_bolus": "#2ecc71"}

# ---------------------------------------------------------------------------
# Figure 1: Box-plot of rel_err per parameter, grouped by config
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

for pi, param in enumerate(params):
    ax = axes[pi]
    data = []
    labels = []
    for cfg in configs:
        vals = [r[f"rel_err_{param}"] for r in rows if r["config"] == cfg]
        data.append(vals)
        labels.append(cfg.replace("_", "\n"))

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6))
    for patch, cfg in zip(bp["boxes"], configs):
        patch.set_facecolor(colors[cfg])
        patch.set_alpha(0.6)

    # Individual points
    for i, cfg in enumerate(configs):
        vals = [r[f"rel_err_{param}"] for r in rows if r["config"] == cfg]
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   color=colors[cfg], alpha=0.5, s=20, zorder=5)

    ax.set_title(param, fontsize=14, fontweight="bold")
    ax.set_ylabel("rel_err")
    ax.set_ylim(bottom=-0.02)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Identification accuracy by parameter and data configuration",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("validation/figures/fig_identifiability_boxplot.png", dpi=150, bbox_inches="tight")
print("Saved: validation/figures/fig_identifiability_boxplot.png")

# ---------------------------------------------------------------------------
# Figure 2: Correlation Vm0-p2U scatter vs rel_err Vm0
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for cfg in configs:
    c_vals = [r["corr_Vm0_p2U"] for r in rows if r["config"] == cfg]
    v_vals = [r["rel_err_Vm0"] for r in rows if r["config"] == cfg]
    ax.scatter(c_vals, v_vals, label=cfg, color=colors[cfg],
               alpha=0.7, s=50, edgecolors="black", linewidth=0.5)

ax.set_xlabel("corr(Vm0, p2U)", fontsize=12)
ax.set_ylabel("rel_err Vm0", fontsize=12)
ax.set_title("Correlation vs Identification Error", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("validation/figures/fig_identifiability_corr.png", dpi=150, bbox_inches="tight")
print("Saved: validation/figures/fig_identifiability_corr.png")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n=== Summary (mean ± std)")
print(f"{'Config':<22} {'kp2':<14} {'Vm0':<14} {'p2U':<14} {'corr(Vm0,p2U)':<14}")
print("-" * 78)
for cfg in configs:
    vals = [r for r in rows if r["config"] == cfg]
    means = {p: np.mean([v[f"rel_err_{p}"] for v in vals]) for p in params}
    stds  = {p: np.std([v[f"rel_err_{p}"] for v in vals]) for p in params}
    corr  = np.mean([v["corr_Vm0_p2U"] for v in vals])
    print(f"{cfg:<22} "
          f"{means['kp2']*100:.1f}±{stds['kp2']*100:.1f}%   "
          f"{means['Vm0']*100:.1f}±{stds['Vm0']*100:.1f}%   "
          f"{means['p2U']*100:.1f}±{stds['p2U']*100:.1f}%   "
          f"{corr:+.3f}")