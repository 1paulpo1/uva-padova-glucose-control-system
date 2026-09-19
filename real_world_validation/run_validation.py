#!/usr/bin/env python3
"""
Staged validation pipeline for T1DEXI/OhioT1DM dataset.

Design:
  Phase 0: scan parquet → filter patients → generate CSVs
  Phase 1: run test_real_data (mode=config) on progressive batches
  Early-stop if gates fail systematically.

Usage:
    python real_world_validation/run_validation.py
    python real_world_validation/run_validation.py --study Loop --batch-start 5 --max-patients 50
    python real_world_validation/run_validation.py --resume validation_results.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from parsers import discover, get_parser_for_source, BaseParser
from data_pipeline import FilterConfig
from batch_runner import StudyScheduler, PipelineOrchestrator

DEFAULT_PARQUET = os.environ.get(
    "UVA_PADOVA_PARQUET",
    os.path.join(os.path.expanduser("~"), "Downloads", "metabonet_public.parquet"),
)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def _make_timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser(
        description="Staged validation on T1DEXI / OhioT1DM")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--study", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--test-real-data", default=None,
                    help="Path to test_real_data")
    ap.add_argument("--resume", default=None,
                    help="Path to existing results JSON (not yet supported in batch_runner)")
    ap.add_argument("--batch-start", type=int, default=5)
    ap.add_argument("--batch-multiplier", type=float, default=2.0)
    ap.add_argument("--max-patients", type=int, default=200)
    ap.add_argument("--min-meals", type=int, default=5)
    ap.add_argument("--min-boluses", type=int, default=5)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--window-stride", type=int, default=1,
                    help="Sliding window stride in days (default: 1)")
    ap.add_argument("--sort-by", default="cgm",
                    choices=["cgm", "meals", "boluses"],
                    help="Sort patients within each study (default: cgm)")
    ap.add_argument("--max-windows-per-patient", type=int, default=0,
                    help="Max sliding windows per patient (0 = unlimited)")
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--timeout", type=int, default=600,
                    help="Max seconds per patient (default: 600, 0=no limit)")
    ap.add_argument("--params-config", default=None,
                    help="Parameter descriptor JSON path (default: from C++ built-in)")
    ap.add_argument("--gate-a-threshold", type=float, default=0.80)
    ap.add_argument("--gate-b-threshold", type=float, default=0.80)
    ap.add_argument("--pass-rate-min", type=float, default=0.20)
    ap.add_argument("--no-cache", action="store_true",
                    help="Force re-scan parquet, ignore cache")
    ap.add_argument("--crash-threshold", type=float, default=0.5)
    ap.add_argument("--min-cgm-density", type=float, default=0.5)
    ap.add_argument("--verbose", action="store_true",
                    help="Show CMA-ES/Ceres output during identification")
    ap.add_argument("--gb-method", default="median",
                    choices=["median", "fasting_p10"])
    ap.add_argument("--min-cgm-points", type=int, default=50)
    ap.add_argument("--max-cgm-gap", type=float, default=120.0)
    ap.add_argument("--outlier-zscore", type=float, default=4.0)
    args = ap.parse_args()

    discover()
    parser_cls = get_parser_for_source(args.parquet)
    if parser_cls is None:
        print(f"Error: no parser found for {args.parquet}")
        sys.exit(1)
    parser = parser_cls()

    script_dir = Path(__file__).parent.resolve()
    output_dir = Path(args.output_dir or script_dir / "datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_real_path = args.test_real_data or (
        script_dir.parent / "build" / "real_world_validation" / "test_real_data")
    test_real_path = Path(test_real_path)
    if not test_real_path.exists():
        print(f"Error: test_real_data not found at {test_real_path}")
        sys.exit(1)

    results_path = output_dir / "validation_results.json"
    csv_summary_path = output_dir / "validation_results.csv"
    analysis_results_path = output_dir / "validation_results_analysis.json"

    thresholds = {
        "gate_a": args.gate_a_threshold,
        "gate_b": args.gate_b_threshold,
        "pass_rate_min": args.pass_rate_min,
        "crash": args.crash_threshold,
    }

    filter_config = FilterConfig(
        gb_method=args.gb_method,
        min_cgm_points=args.min_cgm_points,
        max_cgm_gap_minutes=args.max_cgm_gap,
        outlier_zscore=args.outlier_zscore,
    )

    # ── Phase 0: schedule ─────────────────────────────────────────────
    print("StudyScheduler: scanning & generating RunSpecs...")
    scheduler = StudyScheduler(parser, args.parquet)
    specs = scheduler.generate(
        study_filter=args.study,
        window_days=args.days,
        stride=args.window_stride,
        sort_by=args.sort_by,
        min_meals=args.min_meals,
        min_boluses=args.min_boluses,
        max_windows_per_patient=args.max_windows_per_patient,
        no_cache=args.no_cache,
        output_dir=str(output_dir),
    )
    if not specs:
        print("No RunSpecs generated — nothing to process.")
        sys.exit(0)

    n_unique = len(set(s.patient_id for s in specs))
    print(f"  Generated {len(specs)} RunSpecs ({n_unique} unique patients)")

    # ── Phase 1: orchestrate ──────────────────────────────────────────
    orchestrator = PipelineOrchestrator(
        parser=parser,
        source_path=args.parquet,
        test_real_data_path=str(test_real_path),
        output_dir=str(output_dir),
        filter_config=filter_config,
        params_config=args.params_config,
        restarts=args.restarts,
        evals=args.evals,
        timeout=args.timeout,
        verbose=args.verbose,
        min_cgm_density=args.min_cgm_density,
    )

    print("\nPhase 1: batch validation...")
    batch_results = orchestrator.run(
        specs=specs,
        batch_start=args.batch_start,
        batch_multiplier=args.batch_multiplier,
        max_patients=args.max_patients,
        results_path=str(results_path),
        thresholds=thresholds,
    )

    # ── Phase 2: envelope ─────────────────────────────────────────────
    print(f"\nPhase 2: building analysis envelope ({len(batch_results)} windows)...")
    param_names = None
    envelope_results = []
    for r in batch_results:
        pid = r.get("id")
        spec = r.get("spec")
        if pid is None or spec is None:
            continue
        suffix = f"o{spec.window_start_days}_{spec.window_days}d"
        json_path = output_dir / f"patient{pid}_{suffix}_result.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    pat = json.load(f)
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                print(f"  WARN: skipped patient {pid} (corrupt JSON: {exc})")
                continue
            pat["patient_id"] = pid
            pat["window_start_days"] = spec.window_start_days
            if param_names is None:
                pn = pat.get("param_names", [])
                if pn:
                    param_names = pn
            envelope_results.append(pat)

    envelope = {"results": envelope_results}
    if param_names:
        envelope["param_names"] = param_names
    with open(analysis_results_path, "w") as f:
        json.dump(envelope, f, indent=2)

    # ── CSV summary ───────────────────────────────────────────────────
    with open(csv_summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "window_start", "error", "cost", "success",
                     "gate_a", "gate_b", "cond_number", "wall_time_s"])
        for r in batch_results:
            spec = r.get("spec")
            w.writerow([
                r.get("id"),
                spec.window_start_days if spec else "",
                r.get("error", ""),
                r.get("cost", ""),
                r.get("success", ""),
                r.get("gate_a", {}).get("passed", ""),
                r.get("gate_b", {}).get("passed", ""),
                r.get("gate_c", {}).get("condition_number", ""),
                r.get("wall_time_s", ""),
            ])

    # ── Final stats ───────────────────────────────────────────────────
    n_total = len(batch_results)
    n_pass = sum(1 for r in batch_results
                 if not r.get("error")
                 and r.get("gate_a", {}).get("passed", False)
                 and r.get("gate_b", {}).get("passed", False))
    n_err = sum(1 for r in batch_results if r.get("error"))
    print()
    print("=" * 62)
    print("  VALIDATION COMPLETE")
    print("=" * 62)
    print(f"  Windows processed: {n_total}")
    print(f"  All gates passed:  {n_pass} ({n_pass/n_total:.0%}" if n_total > 0 else "  All gates passed:  0")
    print(f"  Errors:            {n_err}")
    print(f"  Results:           {results_path}")
    print(f"  CSV summary:       {csv_summary_path}")
    print(f"  Analysis envelope: {analysis_results_path} ({len(envelope_results)} windows)")
    print()


if __name__ == "__main__":
    main()
