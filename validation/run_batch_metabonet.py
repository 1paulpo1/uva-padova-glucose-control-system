#!/usr/bin/env python3
"""
Batch validation: run parse_metabonet.py on N patients from the T1DEXI parquet.

Usage:
    python validation/run_batch_metabonet.py [N_patients] [options]

Examples:
    # Report: scan all studies for patients with meals + boluses
    python validation/run_batch_metabonet.py --report-only

    # Report: only OhioT1DM or Loop patients
    python validation/run_batch_metabonet.py --report-only --study OhioT1DM
    python validation/run_batch_metabonet.py --report-only --study Loop

    # Generate CSVs for 10 patients with at least 5 meals and 5 boluses
    python validation/run_batch_metabonet.py 10 --min-meals 5 --min-boluses 5

    # Generate CSVs from a specific study
    python validation/run_batch_metabonet.py 10 --study Loop --min-meals 3 --min-boluses 3
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict

import pyarrow.parquet as pq


DEFAULT_PARQUET = os.environ.get(
    "UVA_PADOVA_PARQUET",
    r"C:\Users\beleg\Downloads\metabonet_public.parquet",
)


# Scanned columns for quick audit
AUDIT_COLS = ["id", "weight", "source_file", "carbs", "bolus", "CGM"]


def scan_patients(parquet_path: str, study: str | None, limit: int | None = None):
    """
    Scan row groups and return dict[pid] = {
        weight_lbs, source_file, rows, meals, boluses, cgm_readings
    }
    If limit is set, stop early once we have that many patients.
    """
    pf = pq.ParquetFile(parquet_path)
    n_groups = pf.metadata.num_row_groups

    # Sample up to 40 row groups for quick scan, or all if limit is small
    n_scan = min(n_groups, max(40, 100 // 1)) if limit is None else n_groups
    if n_scan > 60:
        n_scan = min(n_scan, 60)  # cap for speed

    patients: dict = {}
    done = False

    for rg_idx in range(n_scan):
        if done:
            break
        t = pf.read_row_group(rg_idx, columns=AUDIT_COLS)
        df = t.to_pandas()

        if study is not None:
            df = df[df["source_file"] == study]
        if df.empty:
            continue

        for pid in df["id"].unique():
            if pid in patients:
                # accumulate
                sub = df[df["id"] == pid]
                p = patients[pid]
                p["rows"] += len(sub)
                p["meals"] += int((sub["carbs"].fillna(0) > 0).sum())
                p["boluses"] += int((sub["bolus"].fillna(0) > 0).sum())
                p["cgm"] += int(sub["CGM"].notna().sum())
                continue

            sub = df[df["id"] == pid]
            w = sub["weight"].dropna()
            if len(w) == 0 or w.iloc[0] <= 0:
                continue  # skip patients with no weight

            p = {
                "id": pid,
                "weight_lbs": w.iloc[0],
                "source_file": sub["source_file"].iloc[0],
                "rows": len(sub),
                "meals": int((sub["carbs"].fillna(0) > 0).sum()),
                "boluses": int((sub["bolus"].fillna(0) > 0).sum()),
                "cgm": int(sub["CGM"].notna().sum()),
            }
            patients[pid] = p

            if limit is not None and len(patients) >= limit:
                done = True
                break

        if (rg_idx + 1) % 20 == 0:
            print(f"  RG {rg_idx+1}/{n_scan}: {len(patients)} patients", flush=True)

    return patients


def quick_check_window(parquet_path: str, patient_id: str, n_days: int,
                       min_meals: int, min_boluses: int) -> tuple[bool, int, int]:
    """
    Read the patient data and check if the first n_days have enough meals/boluses.
    Returns (passes, meals_in_window, boluses_in_window).
    """
    try:
        t = pq.read_table(
            parquet_path,
            columns=["id", "date", "carbs", "bolus"],
            filters=[("id", "=", patient_id)],
        )
    except Exception:
        return (False, 0, 0)

    if len(t) == 0:
        return (False, 0, 0)

    df = t.to_pandas()
    df["date"] = df["date"].dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)

    ref = df["date"].iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
    t_end = n_days * 1440
    df["t_min"] = (df["date"] - ref).dt.total_seconds() / 60
    window = df[(df["t_min"] >= 0) & (df["t_min"] < t_end)]

    meals = int((window["carbs"].fillna(0) > 0).sum())
    boluses = int((window["bolus"].fillna(0) > 0).sum())
    return (meals >= min_meals and boluses >= min_boluses, meals, boluses)


def check_csv_quality(cgm_csv: str, basal_csv: str) -> dict:
    """Check data quality of generated CSV files."""
    result = dict(cgm_count=0, meal_count=0, bolus_count=0,
                  basal_segments=0, total_min=0, has_data=False)

    if not os.path.exists(cgm_csv) or not os.path.exists(basal_csv):
        return result

    with open(cgm_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(float(row["t_min"]))
            cgm = float(row["cgm_mgdl"])
            meal = float(row["meal_mg"])
            bolus = float(row["bolus_pmol_kg"])
            if cgm > 0:
                result["cgm_count"] += 1
            if meal > 0:
                result["meal_count"] += 1
            if bolus > 0:
                result["bolus_count"] += 1
            if t > result["total_min"]:
                result["total_min"] = t

    with open(basal_csv, newline="") as f:
        result["basal_segments"] = sum(1 for _ in csv.DictReader(f))

    result["has_data"] = result["cgm_count"] > 0
    return result


def do_report(patients: dict, output_file: str):
    """Print and save full audit report."""
    sorted_patients = sorted(patients.values(), key=lambda p: p["id"])

    fieldnames = ["id", "source_file", "weight_lbs", "rows",
                  "cgm", "meals", "boluses"]
    with open(output_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted_patients)

    # Print summary per study
    by_study = defaultdict(list)
    for p in sorted_patients:
        by_study[p["source_file"]].append(p)

    print(f"\n{'Study':>25} {'Pts':>5} {'With CGM':>10} {'With Meals':>12} "
          f"{'With Boluses':>14} {'Avg Meals':>10} {'Avg Bolus':>10}")
    print("-" * 90)
    for src in sorted(by_study.keys()):
        gp = by_study[src]
        n = len(gp)
        n_cgm = sum(1 for p in gp if p["cgm"] > 0)
        n_meal = sum(1 for p in gp if p["meals"] > 0)
        n_bolus = sum(1 for p in gp if p["boluses"] > 0)
        avg_meal = sum(p["meals"] for p in gp) / n
        avg_bolus = sum(p["boluses"] for p in gp) / n
        print(f"{src:>25} {n:>5} {n_cgm:>10} {n_meal:>12} {n_bolus:>14} "
              f"{avg_meal:>10.1f} {avg_bolus:>10.1f}")

    print(f"\nFull report saved to {output_file}")


def do_generate(patients: list, args):
    """Generate CSVs and run pipeline for selected patients."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    parse_script = args.parse_script or os.path.join(script_dir, "parse_metabonet.py")

    print(f"\nProcessing {len(patients)} patients, {args.days} days each\n")

    results = []
    success_count = 0
    skip_count = 0

    for pid, weight_lbs in patients:
        weight_kg = weight_lbs * 0.453592
        print(f"[{pid}] weight={weight_lbs:.1f} lbs = {weight_kg:.1f} kg ... ",
              end="", flush=True)

        cmd = [
            sys.executable, parse_script, pid,
            "--parquet", args.parquet,
            "--days", str(args.days),
            "--output-dir", output_dir,
            "--weight", f"{weight_kg:.1f}",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if proc.returncode != 0:
            print(f"SKIP ({proc.stderr.strip()})")
            skip_count += 1
            results.append(dict(id=pid, weight_lbs=weight_lbs, weight_kg=weight_kg,
                                status="skip", cgm=0, meals=0, boluses=0,
                                basal_segments=0, days=0))
            continue

        suffix = f"{args.days}d"
        cgm_csv = os.path.join(output_dir, f"patient{pid}_{suffix}_cgm_meals_bolus.csv")
        basal_csv = os.path.join(output_dir, f"patient{pid}_{suffix}_basal.csv")

        quality = check_csv_quality(cgm_csv, basal_csv)

        if not quality["has_data"]:
            print("SKIP (no CGM data)")
            skip_count += 1
            results.append(dict(id=pid, weight_lbs=weight_lbs, weight_kg=weight_kg,
                                status="no_cgm", cgm=0, meals=0, boluses=0,
                                basal_segments=quality["basal_segments"], days=0))
            for p in [cgm_csv, basal_csv]:
                if os.path.exists(p):
                    os.remove(p)
            continue

        ndays = quality["total_min"] // 1440 + 1
        print(f"OK ({quality['cgm_count']} CGM, {quality['meal_count']} meals, "
              f"{quality['bolus_count']} boluses, {quality['basal_segments']} basal segs)")

        success_count += 1
        results.append(dict(id=pid, weight_lbs=weight_lbs, weight_kg=weight_kg,
                            status="ok", cgm=quality["cgm_count"],
                            meals=quality["meal_count"], boluses=quality["bolus_count"],
                            basal_segments=quality["basal_segments"], days=ndays))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary: {success_count} OK, {skip_count} skipped out of {len(patients)}")
    print(f"{'=' * 60}")

    summary_path = os.path.join(output_dir, "batch_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "weight_lbs", "weight_kg", "status",
                                           "cgm", "meals", "boluses", "basal_segments", "days"])
        w.writeheader()
        w.writerows(results)
    print(f"Summary saved to {summary_path}")

    print(f"\n{'ID':>8} {'Wt(lbs)':>8} {'Wt(kg)':>8} {'Status':>10} {'CGM':>8} "
          f"{'Meals':>8} {'Boluses':>8} {'Basal':>8} {'Days':>6}")
    print("-" * 85)
    for r in results:
        print(f"{r['id']:>8} {r['weight_lbs']:>8.1f} {r['weight_kg']:>8.2f} "
              f"{r['status']:>10} {r['cgm']:>8} {r['meals']:>8} {r['boluses']:>8} "
              f"{r['basal_segments']:>8} {r['days']:>6}")


def main():
    ap = argparse.ArgumentParser(
        description="Batch validation on T1DEXI / OhioT1DM parquet")
    ap.add_argument("n_patients", nargs="?", type=int, default=10,
                    help="Number of patients to process")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET,
                    help="Path to parquet file")
    ap.add_argument("--study", type=str, default=None,
                    help="Filter by source_file study (e.g. Loop, DCLP3, OhioT1DM)")
    ap.add_argument("--min-meals", type=int, default=1,
                    help="Minimum meals (default: 1)")
    ap.add_argument("--min-boluses", type=int, default=1,
                    help="Minimum boluses (default: 1)")
    ap.add_argument("--max-scan", type=int, default=2000,
                    help="Max patients to scan before giving up (default: 2000)")
    ap.add_argument("--report-only", action="store_true",
                    help="Only print audit report, don't generate CSVs")
    ap.add_argument("--days", type=int, default=7,
                    help="Days per patient (default: 7)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: validation/output)")
    ap.add_argument("--parse-script", type=str, default=None,
                    help="Path to parse_metabonet.py (default: auto)")
    args = ap.parse_args()

    if not os.path.exists(args.parquet):
        print(f"Error: parquet not found at {args.parquet}")
        sys.exit(1)

    # ---- Scan patients ----
    print("Scanning parquet for patients...")
    scan_limit = None if args.report_only else args.max_scan * 10
    all_patients = scan_patients(args.parquet, args.study, scan_limit)

    if not all_patients:
        print("No patients with weight found!")
        sys.exit(1)

    print(f"  Found {len(all_patients)} patients with weight")

    if args.report_only:
        report_path = os.path.join(
            args.output_dir or os.path.dirname(os.path.abspath(__file__)),
            "patient_audit.csv")
        do_report(all_patients, report_path)
        return

    # ---- Filter by min-meals / min-boluses ----
    # First pass: quick scan (global totals)
    candidates = []
    for pid, p in all_patients.items():
        if p["meals"] >= args.min_meals and p["boluses"] >= args.min_boluses:
            candidates.append((pid, p["weight_lbs"]))

    print(f"  Global scan: {len(candidates)} / {len(all_patients)} have "
          f"meals>={args.min_meals}, boluses>={args.min_boluses}")

    if not candidates:
        print("No patients meet the criteria! Try lower --min-meals/--min-boluses.")
        sys.exit(1)

    # Second pass: check actual N-day window for each candidate
    import random
    random.seed(42)
    random.shuffle(candidates)

    eligible = []
    checked = 0
    for pid, w_lbs in candidates:
        if len(eligible) >= args.n_patients:
            break
        if checked >= args.max_scan:
            break
        checked += 1
        passes, meals_w, boluses_w = quick_check_window(
            args.parquet, pid, args.days, args.min_meals, args.min_boluses)
        if passes:
            eligible.append((pid, w_lbs))
            print(f"  OK {pid}: {meals_w} meals, {boluses_w} boluses in {args.days}d window")
        else:
            print(f"  -- {pid}: only {meals_w} meals, {boluses_w} boluses in {args.days}d window")

    print(f"\n  Window check: {len(eligible)} / {checked} candidates pass "
          f"(scanned {checked}, need {args.n_patients})")

    if not eligible:
        print("No patients pass the window check!")
        sys.exit(1)

    selected = eligible[:args.n_patients]

    # ---- Generate CSVs ----
    do_generate(selected, args)


if __name__ == "__main__":
    main()
