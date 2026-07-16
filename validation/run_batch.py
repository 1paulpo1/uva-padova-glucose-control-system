#!/usr/bin/env python3
"""
Batch validation: parse patient data from parquet/CSV sources into pipeline format.

Usage:
    python validation/run_batch.py scan --source <path> [--study Loop] [--report-only]
    python validation/run_batch.py parse --source <path> --days 7 [--study Loop]
    python validation/run_batch.py check --cgm-csv <path> --basal-csv <path>

Subcommands:
  scan      Scan a dataset and report available patients
  parse     Generate pipeline CSVs for eligible patients
  check     Check quality of already-generated CSVs

Examples:
    python validation/run_batch.py scan --source data.parquet --report-only
    python validation/run_batch.py scan --source data.parquet --study Loop
    python validation/run_batch.py parse --source data.parquet --days 7 \\
        --min-meals 5 --min-boluses 5 --n-patients 20
    python validation/run_batch.py parse --source data.parquet \\
        --pid 559 --pid 563
    python validation/run_batch.py check \\
        --cgm-csv output/patient559_7d_cgm_meals_bolus.csv \\
        --basal-csv output/patient559_7d_basal.csv
"""

import argparse
import os
import random
import sys

from parsers import discover, get_parser_for_source, BaseParser


def _resolve_parser(source: str) -> BaseParser:
    discover()
    cls = get_parser_for_source(source)
    if cls is None:
        print(f"Error: no parser found for source: {source}")
        sys.exit(1)
    return cls()


def cmd_scan(args):
    parser = _resolve_parser(args.source)

    print(f"Scanning: {args.source}")
    if args.study:
        print(f"  Filtering study: {args.study}")

    sig = None if args.no_cache else BaseParser._parquet_signature(args.source)
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "datasets")
    patients = (None if args.no_cache
                else BaseParser._load_scan_cache(output_dir, sig, args.study))

    if patients is None:
        patients = parser.scan(args.source, args.study)
        if not patients:
            print("  No patients with weight found")
            return
        BaseParser._save_scan_cache(output_dir, patients, sig, args.study)
        print(f"  Found {len(patients)} patients with weight")
    else:
        print(f"  Cache hit: {len(patients)} patients with weight")

    if args.report_only:
        out_path = None
        if output_dir:
            out_path = os.path.join(output_dir, "patient_audit.csv")
        BaseParser.report_by_study(patients, out_path)
        return

    # Filter by min meals/boluses
    candidates = [(pid, p["weight_lbs"])
                  for pid, p in patients.items()
                  if p["meals"] >= args.min_meals and p["boluses"] >= args.min_boluses]
    print(f"  Candidates with meals>={args.min_meals}, boluses>={args.min_boluses}: "
          f"{len(candidates)} / {len(patients)}")

    # Window check
    eligible = []
    for pid, w_lbs in candidates:
        passes, meals_w, boluses_w, cgm_w = parser.check_window(
            args.source, pid, args.days, args.min_meals, args.min_boluses)
        if passes:
            eligible.append((pid, w_lbs))

    print(f"  Eligible ({args.days}d window): {len(eligible)} / {len(candidates)} patients")
    return eligible


def cmd_parse(args):
    parser = _resolve_parser(args.source)

    if args.pid:
        patient_ids = args.pid
    else:
        eligible = cmd_scan(args)
        if not eligible:
            print("No eligible patients found")
            return
        random.seed(42)
        random.shuffle(eligible)
        n = min(len(eligible), args.n_patients)
        eligible = eligible[:n]
        patient_ids = [pid for pid, _ in eligible]

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "datasets")

    print(f"\nProcessing {len(patient_ids)} patients, {args.days} days each\n")
    results = []
    for pid in patient_ids:
        print(f"[{pid}] parsing...", end=" ", flush=True)
        try:
            data = parser.parse_patient(pid, args.source, days=args.days,
                                        start_date=args.date)
        except ValueError as e:
            print(f"SKIP ({e})")
            results.append(dict(id=pid, status="skip", error=str(e)))
            continue

        cgm_csv, basal_csv = BaseParser.write_csvs(data, output_dir)
        quality = BaseParser.check_csv_quality(cgm_csv, basal_csv)

        if not quality["has_data"]:
            print("  SKIP (no CGM data)")
            for p in [cgm_csv, basal_csv]:
                if os.path.exists(p):
                    os.remove(p)
            results.append(dict(id=pid, status="no_data"))
            continue

        print(f"OK ({quality['cgm_count']} CGM, {quality['meal_count']} meals)")
        results.append(dict(id=pid, status="ok",
                            cgm=quality["cgm_count"], meals=quality["meal_count"],
                            boluses=quality["bolus_count"],
                            basal_segments=quality["basal_segments"]))

    print(f"\n{'=' * 60}")
    print(f"Summary: {sum(1 for r in results if r['status'] == 'ok')} OK, "
          f"{sum(1 for r in results if r['status'] != 'ok')} skipped")
    print(f"{'=' * 60}")


def cmd_check(args):
    quality = BaseParser.check_csv_quality(args.cgm_csv, args.basal_csv)
    print(f"Quality check results:")
    for key, val in quality.items():
        print(f"  {key}: {val}")


def main():
    ap = argparse.ArgumentParser(description="Batch data parsing for UVA-Padova pipeline")
    sub = ap.add_subparsers(dest="command", required=True)

    # ── scan ──
    p_scan = sub.add_parser("scan", help="Scan dataset and report patients")
    p_scan.add_argument("--source", required=True, help="Path to parquet file")
    p_scan.add_argument("--study", default=None, help="Filter by study (e.g. Loop)")
    p_scan.add_argument("--report-only", action="store_true",
                        help="Only print audit report, don't filter by window")
    p_scan.add_argument("--days", type=int, default=7, help="Window length in days")
    p_scan.add_argument("--min-meals", type=int, default=1)
    p_scan.add_argument("--min-boluses", type=int, default=1)
    p_scan.add_argument("--no-cache", action="store_true")
    p_scan.add_argument("--output-dir", default=None)
    p_scan.set_defaults(func=cmd_scan)

    # ── parse ──
    p_parse = sub.add_parser("parse", help="Generate pipeline CSVs for patients")
    p_parse.add_argument("--source", required=True, help="Path to parquet file")
    p_parse.add_argument("--study", default=None, help="Filter by study")
    p_parse.add_argument("--pid", action="append", help="Specific patient ID(s) to parse")
    p_parse.add_argument("--n-patients", type=int, default=10,
                         help="Number of patients if --pid not given")
    p_parse.add_argument("--days", type=int, default=7)
    p_parse.add_argument("--date", default=None, help="Start date (YYYY-MM-DD)")
    p_parse.add_argument("--min-meals", type=int, default=5)
    p_parse.add_argument("--min-boluses", type=int, default=5)
    p_parse.add_argument("--output-dir", default=None)
    p_parse.add_argument("--no-cache", action="store_true")
    p_parse.set_defaults(func=cmd_parse)

    # ── check ──
    p_check = sub.add_parser("check", help="Check quality of existing CSVs")
    p_check.add_argument("--cgm-csv", required=True)
    p_check.add_argument("--basal-csv", required=True)
    p_check.set_defaults(func=cmd_check)

    args = ap.parse_args()
    args._parser = ap  # pass parser to subcommands for --help
    args.func(args)


if __name__ == "__main__":
    main()
