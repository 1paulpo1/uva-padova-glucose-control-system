#!/usr/bin/env python3
"""
Visualise Gate (a): restart-point clusters in parameter space.

Reads gate_restarts.csv produced by debug_fim and plots 3D scatter +
pairwise projections coloured by cluster label.

Usage:
    python3 validation/plot_gates.py [csv_path] [figures_dir]

Default csv_path:    validation/output/gate_restarts.csv
Default figures_dir: validation/figures/
"""

import os
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "output", "gate_restarts.csv")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

if len(sys.argv) > 1:
    CSV_PATH = sys.argv[1]
if len(sys.argv) > 2:
    FIGURES_DIR = sys.argv[2]

os.makedirs(FIGURES_DIR, exist_ok=True)

# ---- Read CSV ----
rows = []
param_names = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    param_names = [k for k in reader.fieldnames if k not in ("type", "cluster")]
    for r in reader:
        rows.append(r)

if not rows:
    print("No data found in", CSV_PATH)
    sys.exit(1)

# Split by type
restarts = [r for r in rows if r["type"] == "restart"]
estimates = [r for r in rows if r["type"] == "estimate"]
truths = [r for r in rows if r["type"] == "truth"]

# Build array
n_params = len(param_names)
X = np.array([[float(r[p]) for p in param_names] for r in restarts])
clusters = np.array([int(r["cluster"]) for r in restarts])

has_estimate = len(estimates) > 0
has_truth = len(truths) > 0
theta_est = np.array([float(estimates[0][p]) for p in param_names]) if has_estimate else None
theta_true = np.array([float(truths[0][p]) for p in param_names]) if has_truth else None

n_clusters = len(set(c for c in clusters if c >= 0))
n_noise = sum(1 for c in clusters if c < 0)

# ---- Colour map ----
cmap = plt.colormaps["tab10"]
colors = [cmap(c % 10) if c >= 0 else (0.5, 0.5, 0.5, 0.5) for c in clusters]

# ---- Figure ----
if n_params == 3:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Restarts
    for i in range(X.shape[0]):
        marker = "o" if clusters[i] >= 0 else "x"
        ax.scatter(*X[i], color=colors[i], s=80, marker=marker, alpha=0.8,
                   label=f"cluster {clusters[i]}" if i == 0 or (i > 0 and clusters[i] != clusters[i-1]) else "")
    # Estimate
    if theta_est is not None:
        ax.scatter(*theta_est, color="black", s=150, marker="D", label="estimate")
    # Truth
    if theta_true is not None:
        ax.scatter(*theta_true, color="red", s=150, marker="*", label="truth")

    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_zlabel(param_names[2])

    title = f"Gate (a): {n_clusters} cluster(s)"
    if n_noise > 0:
        title += f", {n_noise} noise"
    if n_clusters <= 1:
        title += " ✓ PASSED"
    else:
        title += " ✗ BLOCKED"
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "gate_a_3d.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

# ---- Pairwise scatter matrix (always) ----
fig, axes = plt.subplots(n_params, n_params, figsize=(3 * n_params, 3 * n_params))
for i in range(n_params):
    for j in range(n_params):
        ax = axes[i, j]
        if i == j:
            # Histogram on diagonal
            for cid in set(clusters):
                mask = clusters == cid
                ax.hist(X[mask, i], bins=6, alpha=0.5,
                        color=cmap(cid % 10) if cid >= 0 else "gray",
                        label=f"cluster {cid}" if cid >= 0 else "noise")
            if theta_est is not None:
                ax.axvline(theta_est[i], color="black", ls="--", lw=1.5, label="estimate")
            if theta_true is not None:
                ax.axvline(theta_true[i], color="red", ls=":", lw=1.5, label="truth")
            ax.set_xlabel(param_names[i])
        else:
            for k in range(X.shape[0]):
                c = colors[k]
                marker = "o" if clusters[k] >= 0 else "x"
                ax.scatter(X[k, j], X[k, i], color=c, s=60, marker=marker, alpha=0.7)
            if theta_est is not None:
                ax.scatter(theta_est[j], theta_est[i], color="black", s=120, marker="D", zorder=5)
            if theta_true is not None:
                ax.scatter(theta_true[j], theta_true[i], color="red", s=120, marker="*", zorder=5)
            ax.set_xlabel(param_names[j])
            ax.set_ylabel(param_names[i])

    n_clusters_str = f"{n_clusters} cluster(s)"
    if n_noise > 0:
        n_clusters_str += f" + {n_noise} noise"
    axes[0, -1].legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", label=n_clusters_str),
            plt.Line2D([0], [0], marker="D", color="black", label="estimate"),
            plt.Line2D([0], [0], marker="*", color="red", label="truth"),
        ],
        fontsize=9, loc="upper left"
    )

fig.suptitle("Gate (a): restart points projected onto parameter planes", fontsize=13, y=1.02)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "gate_a_pairwise.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
print(f"Saved {path}")
