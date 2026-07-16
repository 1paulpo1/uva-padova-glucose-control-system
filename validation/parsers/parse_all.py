#!/usr/bin/env python3
"""
Parse ALL patients from parquet, filter by quality, group by study.

Output: validation/datasets/<study_name>/patient{id}_{ndays}d_*.csv

Usage:
    python -m parsers.parse_all
    python -m parsers.parse_all --parquet <path> --out validation/datasets/
    python -m parsers.parse_all --days 7 --limit 100 --no-cache
"""

import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import pyarrow

from . import discover, get_parser_for_source
from .base_parser import BaseParser

DEFAULT_PARQUET = os.environ.get(
    "UVA_PADOVA_PARQUET",
    r"C:\Users\beleg\Downloads\metabonet_public.parquet",
)


def main():
    ap = argparse.ArgumentParser(
        description="Parse all patients from parquet, filter by quality, group by study")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--study", default=None,
                    help="Filter by study name (e.g. Flair, Loop)")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: validation/datasets/)")
    ap.add_argument("--days", type=int, default=0,
                    help="Window in days (0 = all available data)")
    ap.add_argument("--min-meals-per-day", type=float, default=1.0)
    ap.add_argument("--min-boluses-per-day", type=float, default=1.0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max patients to process (0 = all)")
    args = ap.parse_args()

    # validation/ (two levels up from this file inside parsers/)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.out or os.path.join(script_dir, "datasets")
    os.makedirs(output_dir, exist_ok=True)

    discover()
    parser_cls = get_parser_for_source(args.parquet)
    if parser_cls is None:
        print(f"Error: no parser for {args.parquet}")
        sys.exit(1)
    parser = parser_cls()

    # Scan all patients (no study filter)
    print("Scanning parquet (all studies)...")
    sig = None if args.no_cache else BaseParser._parquet_signature(args.parquet)
    all_pts = (None if args.no_cache
               else BaseParser._load_scan_cache(output_dir, sig, None))

    if all_pts is None:
        t0 = time.time()
        all_pts = parser.scan(args.parquet, study_filter=None, limit=0)
        if not all_pts:
            print("No patients found")
            sys.exit(1)
        BaseParser._save_scan_cache(output_dir, all_pts, sig, None)
        print(f"  Found {len(all_pts)} patients in {time.time() - t0:.1f}s")
    else:
        print(f"  Cache hit: {len(all_pts)} patients")

    # Group by study
    by_study = defaultdict(list)
    for pid, p in all_pts.items():
        by_study[p["source_file"]].append((pid, p))

    print(f"\n{'Study':>25} {'Patients':>10}")
    print("-" * 40)
    for src in sorted(by_study):
        print(f"{src:>25} {len(by_study[src]):>10}")
    print()

    # Parse per study group
    patient_count = 0
    total_ok = 0
    total_skip = 0
    records = []
    max_patients = args.limit if args.limit > 0 else None

    for study_name in sorted(by_study):
        if args.study and study_name != args.study:
            continue
        if max_patients is not None and patient_count >= max_patients:
            break
        patients = by_study[study_name]
        study_dir = os.path.join(output_dir, study_name)

        for pid, p in patients:
            if max_patients is not None and patient_count >= max_patients:
                break
            patient_count += 1

            if p.get("weight_lbs", 0) <= 0:
                print(f"  [{pid}] SKIP (no weight)")
                records.append(dict(id=pid, study=study_name,
                                    status="skip", reason="no weight"))
                total_skip += 1
                continue

            # Resume: skip if CSVs already exist
            existing = glob.glob(os.path.join(study_dir, f"patient{pid}_*_cgm_meals_bolus.csv"))
            if existing:
                print(f"  [{pid}] {study_name}... SKIP (already done)")
                records.append(dict(id=pid, study=study_name, status="skip", reason="already_done"))
                total_skip += 1
                continue

            print(f"  [{pid}] {study_name}...", end=" ", flush=True)
            try:
                data = parser.parse_patient(pid, args.parquet, days=args.days)
            except (ValueError, OSError, pyarrow.ArrowException) as e:
                print(f"SKIP ({e})")
                records.append(dict(id=pid, study=study_name,
                                    status="skip", reason=str(e)))
                total_skip += 1
                continue

            ndays = max(1, data.total_min / (24 * 60))
            meals_per_day = data.n_meals / ndays
            boluses_per_day = data.n_boluses / ndays

            if data.n_basal_segments == 0:
                print("SKIP (no basal segments)")
                records.append(dict(id=pid, study=study_name, status="skip",
                                    reason="no_basal"))
                total_skip += 1
                continue
            if meals_per_day < args.min_meals_per_day:
                print(f"SKIP ({data.n_meals} meals / {ndays:.1f}d = {meals_per_day:.1f}/d)")
                records.append(dict(id=pid, study=study_name, status="skip",
                                    reason=f"meals {meals_per_day:.1f}/d"))
                total_skip += 1
                continue
            if boluses_per_day < args.min_boluses_per_day:
                print(f"SKIP ({data.n_boluses} boluses / {ndays:.1f}d = {boluses_per_day:.1f}/d)")
                records.append(dict(id=pid, study=study_name, status="skip",
                                    reason=f"boluses {boluses_per_day:.1f}/d"))
                total_skip += 1
                continue

            ndays_int = data.total_min // (24 * 60)
            suffix = f"{ndays_int}d"
            BaseParser.write_csvs(data, study_dir, suffix=suffix)
            print(f"OK ({data.n_cgm} CGM, {data.n_meals} meals, "
                  f"{data.n_basal_segments} basal, {ndays_int}d)")
            records.append(dict(id=pid, study=study_name, status="ok",
                                cgm=data.n_cgm, meals=data.n_meals,
                                boluses=data.n_boluses,
                                basal_segments=data.n_basal_segments,
                                ndays=ndays_int))
            total_ok += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  DONE: {total_ok} OK, {total_skip} skipped / {patient_count} total")
    print(f"{'=' * 60}")

    by_study_summary = defaultdict(lambda: dict(ok=0, skip=0))
    for r in records:
        by_study_summary[r["study"]][r["status"]] += 1

    print(f"\n{'Study':>25} {'OK':>6} {'Skip':>6}")
    print("-" * 40)
    for src in sorted(by_study_summary):
        s = by_study_summary[src]
        print(f"{src:>25} {s['ok']:>6} {s['skip']:>6}")

    skip_reasons = defaultdict(int)
    for r in records:
        if r["status"] == "skip":
            skip_reasons[r["reason"]] += 1
    if skip_reasons:
        print(f"\nSkip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print()


if __name__ == "__main__":
    main()
