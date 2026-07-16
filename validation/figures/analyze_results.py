#!/usr/bin/env python3
"""
Analyze validation results from test_real_data.exe JSON output.

Re-exports all ground-truth-free plot functions from crossval/analyze_results.py.
Usage:
    python -m figures.analyze_results --json output/validation_results_analysis.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Re-export all reusable functions from crossval
_crossval = Path(__file__).resolve().parent.parent.parent / "crossval"
if str(_crossval) not in sys.path:
    sys.path.insert(0, str(_crossval))

from analyze_results import (
    load_results,
    precompute,
    filter_results,
    PLOT_REGISTRY,
    plot_gate_summary,
    plot_pca_tsne,
    plot_cost_vs_clusters,
    plot_residual_timeseries,
    plot_residual_acf,
    plot_cmaes_landscape,
    plot_evals_histogram,
    plot_cost_improvement,
    plot_prior_dominated_bar,
    print_detailed_gate_table,
    print_prior_dominated_ranking,
    print_summary_table,
)


def main():
    ap = argparse.ArgumentParser(
        description="Analyze validation results from test_real_data.exe")
    ap.add_argument("--json", default=None,
                    help="Path to analysis envelope JSON")
    ap.add_argument("--out", default=None,
                    help="Output directory for figures")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip plot generation")
    ap.add_argument("--plot", nargs="+", default=None,
                    help="Specific plots to generate (default: all)")
    ap.add_argument("--gate-a", action="store_true",
                    help="Filter to Gate A passes only")
    ap.add_argument("--gate-b", action="store_true",
                    help="Filter to Gate B passes only")
    ap.add_argument("--include-failures", action="store_true",
                    help="Include failure cases in gate summary plots")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Resolve JSON path
    script_dir = Path(__file__).resolve().parent  # validation/figures/
    output_dir = Path(args.out or script_dir.parent / "datasets")
    json_path = args.json or str(output_dir / "validation_results_analysis.json")

    if not os.path.exists(json_path):
        print(f"Error: analysis JSON not found at {json_path}")
        print("Run validation/run_validation.py first to generate results.")
        sys.exit(1)

    data = load_results(json_path)
    n_params = len(data.get("param_names", []))
    if n_params == 0:
        print("Warning: param_names is empty")

    # Preserve raw results
    data["results_raw"] = data["results"]

    # Build raw DataFrame
    data["raw_df"] = precompute(data["results_raw"], n_params)

    # Build inclusive DataFrame if requested
    if args.include_failures:
        inclu = [r for r in data["results"] if r.get("success")]
        data["inclusive_df"] = precompute(inclu, n_params) if inclu else None
    else:
        data["inclusive_df"] = None

    # Filter
    filtered = filter_results(
        data["results"],
        gate_a=args.gate_a,
        gate_b=args.gate_b)

    data["results"] = filtered
    data["df"] = precompute(filtered, n_params)

    print(f"Loaded {json_path}")
    print(f"  Total patients:     {len(data['results_raw'])}")
    print(f"  Filtered patients:  {len(filtered)}")
    print(f"  Parameters:         {n_params}")

    # Print tables
    print("\n-- Detailed Gate Table --")
    print_detailed_gate_table(data)

    print("\n-- Prior-Dominated Ranking --")
    print_prior_dominated_ranking(data)

    print("\n-- Summary Table --")
    print_summary_table(data)

    # Generate plots
    if not args.no_plots:
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        plot_keys = args.plot or list(PLOT_REGISTRY.keys())
        for key in plot_keys:
            if key not in PLOT_REGISTRY:
                print(f"  Skipping unknown plot '{key}'")
                continue
            fn = PLOT_REGISTRY[key]
            if args.verbose:
                print(f"  Plot: {key}...")
            try:
                fn(data, str(fig_dir))
            except Exception as e:
                print(f"  Error plotting {key}: {e}")

        print(f"\nFigures saved to {fig_dir}")


if __name__ == "__main__":
    main()
