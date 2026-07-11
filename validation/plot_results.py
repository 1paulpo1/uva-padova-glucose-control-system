#!/usr/bin/env python3
"""
Plot validation results from run_scenarios CSV output.

Usage:
    python3 validation/plot_results.py [output_dir] [figures_dir]

Default output_dir: uva-padova-cpp/validation/output/
Default figures_dir: uva-padova-cpp/validation/figures/
"""

import os
import sys
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

if len(sys.argv) > 1:
    OUTPUT_DIR = sys.argv[1]
if len(sys.argv) > 2:
    FIGURES_DIR = sys.argv[2]

os.makedirs(FIGURES_DIR, exist_ok=True)

# ---- Style ----
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.8,
    "lines.linewidth": 1.5,
})


def load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return None, None
    with open(path) as f:
        reader = csv.DictReader(f)
        data = list(reader)
    headers = reader.fieldnames
    # Convert to dict of arrays
    cols = {h: np.array([float(r[h]) for r in data]) for h in headers}
    print(f"  Loaded {path} ({len(data)} rows, {len(headers)} cols)")
    return data, cols


# ==== Figure 1: Steady-state hold ====
def plot_ss_hold():
    print("\n--- Figure 1: Steady-state hold ---")
    data, cols = load_csv("ss_hold.csv")
    if cols is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    t = cols["t_min"] / 60.0  # convert to hours

    ax = axes[0]
    ax.plot(t, cols["G"], label="Plasma glucose (G)", color="tab:red")
    ax.plot(t, cols["Gsc"], label="CGM (Gsc)", color="tab:orange", ls="--")
    ax.axhline(y=120, color="gray", ls=":", alpha=0.5, label="Gb = 120")
    ax.set_ylabel("Glucose [mg/dL]")
    ax.legend(loc="best")
    ax.set_title("Steady-State Hold (360 min)")

    ax = axes[1]
    ax.plot(t, cols["Ip"], label="Plasma insulin (Ip)", color="tab:blue")
    ax.plot(t, cols["Ii"], label="Hepatic insulin (Ii)", color="tab:cyan", ls="--")
    ax.set_ylabel("Insulin [pmol/kg]")
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(t, cols["Isc1"], label="SC fast (Isc1)", color="tab:green")
    ax.plot(t, cols["Isc2"], label="SC slow (Isc2)", color="tab:olive", ls="--")
    ax.set_ylabel("SC Insulin [pmol/kg]")
    ax.set_xlabel("Time [hours]")
    ax.legend(loc="best")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure_01_steady_state.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ==== Figure 2: Gastric emptying curve ====
def plot_gastric_emptying():
    print("\n--- Figure 2: Gastric emptying curve ---")
    data, cols = load_csv("gastric_emptying.csv")
    if cols is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    D_values = sorted(set(cols["D"]))
    for D in D_values:
        mask = cols["D"] == D
        q = cols["Qsto"][mask] / 1000.0  # convert to g
        k = cols["k_emp"][mask]
        label = f"D = {D/1000:.0f}g" if D > 0 else "D = 0 (no meal)"
        ax.plot(q, k, label=label)

    ax.set_xlabel("Stomach content Qsto [g]")
    ax.set_ylabel("Gastric emptying rate k_emp [1/min]")
    ax.set_title("Gastric Emptying Rate vs Stomach Content")
    ax.legend(loc="best")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure_02_gastric_emptying.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ==== Figure 3: 60-min meal response ====
def plot_meal_60min():
    print("\n--- Figure 3: 60-min meal response ---")
    data, cols = load_csv("meal_60min.csv")
    if cols is None:
        return

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    t = cols["t_min"]

    ax = axes[0]
    ax.plot(t, cols["G"], label="G (plasma)", color="tab:red", lw=2)
    ax.plot(t, cols["Gsc"], label="Gsc (CGM)", color="tab:orange", ls="--")
    ax.axhline(y=120, color="gray", ls=":", alpha=0.5, label="Gb = 120")
    ax.set_ylabel("Glucose [mg/dL]")
    ax.legend(loc="best")
    ax.set_title("60-min Postprandial Response (50g CHO, no bolus)")

    ax = axes[1]
    ax.plot(t, cols["Qsto1"], label="Qsto1 (solid)", color="tab:brown")
    ax.plot(t, cols["Qsto2"], label="Qsto2 (liquid)", color="tab:orange")
    ax.plot(t, cols["Qgut"], label="Qgut (gut)", color="tab:green")
    ax.set_ylabel("GI Content [mg]")
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(t, cols["Ra_meal"], label="Ra_meal", color="tab:purple")
    ax.plot(t, cols["EGP"], label="EGP", color="tab:red", ls="--")
    ax.plot(t, cols["Uid"], label="Uid (insulin-dep. util.)", color="tab:blue", ls=":")
    ax.set_ylabel("Flux [mg/kg/min]")
    ax.legend(loc="best")

    ax = axes[3]
    ax.plot(t, cols["Ip"], label="Ip (plasma)", color="tab:blue")
    ax.plot(t, cols["Ii"], label="Ii (hepatic)", color="tab:cyan", ls="--")
    ax.set_ylabel("Insulin [pmol/kg]")
    ax.set_xlabel("Time [min]")
    ax.legend(loc="best")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure_03_meal_response.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ==== Figure 4: 24h profile ====
def plot_24h():
    print("\n--- Figure 4: 24h profile ---")
    data, cols = load_csv("day_24h.csv")
    if cols is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    t = cols["t_min"] / 60.0  # hours

    # Meal markers
    meal_times = []
    meal_sizes = []
    for i, flag in enumerate(cols["meal_flag"]):
        if flag > 0:
            meal_times.append(cols["t_min"][i] / 60.0)
            meal_sizes.append(flag / 1000.0)  # g

    ax = axes[0]
    ax.plot(t, cols["G"], label="G (plasma)", color="tab:red", lw=2)
    ax.plot(t, cols["Gsc"], label="Gsc (CGM)", color="tab:orange", ls="--")
    ax.axhline(y=120, color="gray", ls=":", alpha=0.5, label="Gb = 120")
    # Meal markers
    for mt, ms in zip(meal_times, meal_sizes):
        ax.axvline(x=mt, color="green", alpha=0.3, lw=2)
        ax.text(mt, ax.get_ylim()[1] * 0.95, f"{ms:.0f}g", fontsize=8,
                ha="center", color="green", alpha=0.7)
    ax.set_ylabel("Glucose [mg/dL]")
    ax.legend(loc="upper right")
    ax.set_title("24-Hour Multi-Meal Simulation")

    ax = axes[1]
    ax.plot(t, cols["Ip"], label="Ip (plasma)", color="tab:blue")
    ax.plot(t, cols["Ii"], label="Ii (hepatic)", color="tab:cyan", ls="--")
    ax.set_ylabel("Insulin [pmol/kg]")
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(t, cols["Qsto1"], label="Qsto1 (solid)", color="tab:brown")
    ax.plot(t, cols["Qsto2"], label="Qsto2 (liquid)", color="tab:orange")
    ax.plot(t, cols["Qgut"], label="Qgut (gut)", color="tab:green")
    for mt in meal_times:
        ax.axvline(x=mt, color="green", alpha=0.2, lw=1)
    ax.set_ylabel("GI Content [mg]")
    ax.set_xlabel("Time [hours]")
    ax.legend(loc="best")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure_04_24h_profile.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ==== Figure 5: Insulin pulse ====
def plot_insulin_pulse():
    print("\n--- Figure 5: Insulin pulse response ---")
    data, cols = load_csv("insulin_pulse.csv")
    if cols is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    t = cols["t_min"]

    ax = axes[0]
    ax.plot(t, cols["Ip"], label="Ip (plasma)", color="tab:blue", lw=2)
    ax.plot(t, cols["Ii"], label="Ii (hepatic)", color="tab:cyan", ls="--")
    ax.axvline(x=0, color="red", alpha=0.3, lw=2, label="Bolus")
    ax.set_ylabel("Insulin [pmol/kg]")
    ax.legend(loc="best")
    ax.set_title("Insulin Pulse Response (0.5 U bolus at t=0)")

    ax = axes[1]
    ax.plot(t, cols["Isc1"], label="Isc1 (SC fast)", color="tab:green")
    ax.plot(t, cols["Isc2"], label="Isc2 (SC slow)", color="tab:olive", ls="--")
    ax.axvline(x=0, color="red", alpha=0.3, lw=2)
    ax.set_ylabel("SC Insulin [pmol/kg]")
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(t, cols["X"], label="X (utilization action)", color="tab:purple")
    ax.plot(t, cols["I_prime"], label="I' (intermediate)", color="tab:orange")
    ax.plot(t, cols["Xl"], label="Xl (EGP delay)", color="tab:red", ls="--")
    ax.axvline(x=0, color="red", alpha=0.3, lw=2)
    ax.set_ylabel("Signals [various units]")
    ax.set_xlabel("Time [min]")
    ax.legend(loc="best")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure_05_insulin_pulse.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ==== Figure 6: Delivery comparison (pulse vs uniform) ====
def plot_comparison_delivery():
    print("\n--- Figure 6: Delivery comparison ---")
    meal_data, meal_cols = load_csv("comparison_meal.csv")
    ins_data, ins_cols = load_csv("comparison_insulin.csv")
    if meal_cols is None or ins_cols is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plt.subplots_adjust(hspace=0.35, wspace=0.30)

    # ---- Row 1: Meal ----
    t = meal_cols["t_min"]

    ax = axes[0, 0]
    ax.plot(t, meal_cols["G_pulse"], label="Pulse (T=1)", color="tab:red", lw=2)
    ax.plot(t, meal_cols["G_uniform"], label="Uniform (T=15)", color="tab:orange", lw=2, ls="--")
    ax.axhline(y=120, color="gray", ls=":", alpha=0.5)
    ax.set_ylabel("Glucose [mg/dL]")
    ax.set_title("G — Meal (50g CHO)")
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(t, meal_cols["Gsc_pulse"], label="Pulse", color="tab:red", lw=2)
    ax.plot(t, meal_cols["Gsc_uniform"], label="Uniform", color="tab:orange", lw=2, ls="--")
    ax.axhline(y=120, color="gray", ls=":", alpha=0.5)
    ax.set_title("Gsc (CGM)")
    ax.legend(loc="best")

    ax = axes[0, 2]
    ax.plot(t, meal_cols["Qsto1_pulse"] / 1000.0, label="Pulse", color="tab:red", lw=2)
    ax.plot(t, meal_cols["Qsto1_uniform"] / 1000.0, label="Uniform", color="tab:orange", lw=2, ls="--")
    ax.set_ylabel("Qsto1 [g]")
    ax.set_title("Stomach Content (Qsto1)")
    ax.legend(loc="best")

    # ---- Row 2: Insulin ----
    t2 = ins_cols["t_min"]

    ax = axes[1, 0]
    ax.plot(t2, ins_cols["Ip_pulse"], label="Pulse (T=1)", color="tab:blue", lw=2)
    ax.plot(t2, ins_cols["Ip_uniform"], label="Uniform (T=30)", color="tab:cyan", lw=2, ls="--")
    ax.axvline(x=0, color="gray", alpha=0.3)
    ax.set_ylabel("Ip [pmol/kg]")
    ax.set_title("Plasma Insulin (Ip) — 0.5U bolus")
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(t2, ins_cols["Isc1_pulse"], label="Isc1 pulse", color="tab:green", lw=2)
    ax.plot(t2, ins_cols["Isc2_pulse"], label="Isc2 pulse", color="tab:olive", lw=2)
    ax.plot(t2, ins_cols["Isc1_uniform"], label="Isc1 uniform", color="tab:green", lw=2, ls="--")
    ax.plot(t2, ins_cols["Isc2_uniform"], label="Isc2 uniform", color="tab:olive", lw=2, ls=":")
    ax.axvline(x=0, color="gray", alpha=0.3)
    ax.set_title("SC Insulin (Isc1/2)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 2]
    ax.plot(t2, ins_cols["X_pulse"], label="X pulse", color="tab:purple", lw=2)
    ax.plot(t2, ins_cols["X_uniform"], label="X uniform", color="tab:purple", lw=2, ls="--")
    ax.axvline(x=0, color="gray", alpha=0.3)
    ax.set_xlabel("Time [min]")
    ax.set_title("Insulin Action (X)")
    ax.legend(loc="best")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure_06_delivery_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ==== Main ====
def main():
    print(f"Plotting from: {OUTPUT_DIR}")
    print(f"Saving to:    {FIGURES_DIR}")

    plot_ss_hold()
    plot_gastric_emptying()
    plot_meal_60min()
    plot_24h()
    plot_insulin_pulse()
    plot_comparison_delivery()

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
