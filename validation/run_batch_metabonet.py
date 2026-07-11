#!/usr/bin/env python3
"""
Batch validation: run parse_metabonet.py on N patients from the T1DEXI parquet.

Generates CSVs and produces a summary of data quality for each patient.

Usage:
    python validation/run_batch_metabonet.py [N_patients] [options]

Examples:
    # Process first 10 patients with valid data
    python validation/run_batch_metabonet.py 10

    # Process first 20 patients, 7 days each
    python validation/run_batch_metabonet.py 20 --days 7 --output-dir validation/output
"""

import argparse
import csv
import os
import random
import subprocess
import sys
from datetime import datetime

import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_PARQUET = os.environ.get(
    "UVA_PADOVA_PARQUET",
    r"C:\Users\beleg\Downloads\metabonet_public.parquet",
)


def collect_patients_with_weight(parquet_path: str, limit: int, study: str | None):
    """
    Scan parquet row groups and collect patient IDs that have weight > 0.
    If study is given, only patients from that source_file.
    Returns a list of (pid, weight_lbs) tuples.
    """
    pf = pq.ParquetFile(parquet_path)
    patients = {}
    n_groups = pf.metadata.num_row_groups

    print(f"Scanning {n_groups} row groups for patients with weight...")
    for rg_idx in range(n_groups):
        t = pf.read_row_group(rg_idx, columns=["id", "weight", "source_file"])
        df = t.to_pandas()
        if study is not None:
            df = df[df["source_file"] == study]
        for pid in df["id"].unique():
            if pid in patients:
                continue
            sub = df[df["id"] == pid]
            w = sub["weight"].dropna()
            if len(w) > 0 and w.iloc[0] > 0:
                patients[pid] = (w.iloc[0], sub["source_file"].iloc[0])
        if (rg_idx + 1) % 30 == 0:
            print(f"  RG {rg_idx+1}/{n_groups}: {len(patients)} patients found")
            if len(patients) >= limit:
                break

    if study is not None:
        result = [(pid, w) for pid, (w, src) in patients.items() if src == study]
    else:
        result = list(patients.items())
        result = [(pid, w) for pid, (w, src) in result]

    # Shuffle for diverse sampling, then take limit
    random.shuffle(result)
    return result[:limit]


def check_csv_quality(cgm_csv: str, basal_csv: str) -> dict:
    """Check data quality of generated CSV files."""
    result = {"cgm_count": 0, "meal_count": 0, "bolus_count": 0,
              "basal_segments": 0, "total_min": 0, "has_data": False}

    if not os.path.exists(cgm_csv) or not os.path.exists(basal_csv):
        return result

    # Check CGM file
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

    # Check basal file
    with open(basal_csv, newline="") as f:
        result["basal_segments"] = sum(1 for _ in csv.DictReader(f))

    result["has_data"] = result["cgm_count"] > 0
    return result


def main():
    ap = argparse.ArgumentParser(description="Batch validation on T1DEXI parquet")
    ap.add_argument("n_patients", nargs="?", type=int, default=10,
                    help="Number of patients to process (default: 10)")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET,
                    help="Path to parquet file")
    ap.add_argument("--days", type=int, default=7,
                    help="Days per patient (default: 7)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: validation/output)")
    ap.add_argument("--parse-script", type=str, default=None,
                    help="Path to parse_metabonet.py (default: auto)")
    ap.add_argument("--study", type=str, default=None,
                    help="Filter by source_file study (e.g. Loop, DCLP3, OhioT1DM)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for patient sampling (default: 42)")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    parse_script = args.parse_script or os.path.join(script_dir, "parse_metabonet.py")
    if not os.path.exists(parse_script):
        print(f"Error: parse script not found at {parse_script}")
        sys.exit(1)

    if not os.path.exists(args.parquet):
        print(f"Error: parquet not found at {args.parquet}")
        sys.exit(1)

    random.seed(args.seed)

    # ---- Collect patients ----
    patients = collect_patients_with_weight(args.parquet, args.n_patients, args.study)
    if not patients:
        print("No patients with weight found!")
        sys.exit(1)

    print(f"\nProcessing {len(patients)} patients, {args.days} days each\n")

    # ---- Process each patient ----
    results = []
    success_count = 0
    skip_count = 0

    for pid, weight_lbs in patients:
        weight_kg = weight_lbs * 0.453592
        print(f"[{pid}] weight={weight_lbs:.1f} lbs = {weight_kg:.1f} kg ... ", end="", flush=True)

        # Run parse_metabonet.py
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
            results.append({
                "id": pid, "weight_lbs": weight_lbs, "weight_kg": weight_kg, "status": "skip",
                "cgm": 0, "meals": 0, "boluses": 0,
                "basal_segments": 0, "days": 0,
            })
            continue

        # Check quality
        suffix = f"{args.days}d"
        cgm_csv = os.path.join(output_dir, f"patient{pid}_{suffix}_cgm_meals_bolus.csv")
        basal_csv = os.path.join(output_dir, f"patient{pid}_{suffix}_basal.csv")

        quality = check_csv_quality(cgm_csv, basal_csv)

        if not quality["has_data"]:
            print("SKIP (no CGM data)")
            skip_count += 1
            results.append({
                "id": pid, "weight_lbs": weight_lbs, "weight_kg": weight_kg, "status": "no_cgm",
                "cgm": 0, "meals": 0, "boluses": 0,
                "basal_segments": quality["basal_segments"],
                "days": 0,
            })
            # Remove useless files
            for p in [cgm_csv, basal_csv]:
                if os.path.exists(p):
                    os.remove(p)
            continue

        ndays = quality["total_min"] // 1440 + 1
        print(f"OK ({quality['cgm_count']} CGM, {quality['meal_count']} meals, "
              f"{quality['bolus_count']} boluses, {quality['basal_segments']} basal segs)")

        success_count += 1
        results.append({
            "id": pid, "weight_lbs": weight_lbs, "weight_kg": weight_kg, "status": "ok",
            "cgm": quality["cgm_count"],
            "meals": quality["meal_count"],
            "boluses": quality["bolus_count"],
            "basal_segments": quality["basal_segments"],
            "days": ndays,
        })

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Summary: {success_count} OK, {skip_count} skipped out of {len(patients)}")
    print(f"{'='*60}")

    summary_path = os.path.join(output_dir, "batch_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "weight_lbs", "weight_kg", "status", "cgm",
                                           "meals", "boluses", "basal_segments", "days"])
        w.writeheader()
        w.writerows(results)
    print(f"\nSummary saved to {summary_path}")

    # Print table
    print(f"\n{'ID':>8} {'Wt(lbs)':>8} {'Wt(kg)':>8} {'Status':>10} {'CGM':>8} {'Meals':>8} "
          f"{'Boluses':>8} {'Basal':>8} {'Days':>6}")
    print("-" * 75)
    for r in results:
        print(f"{r['id']:>8} {r['weight_lbs']:>8.1f} {r['weight_kg']:>8.2f} {r['status']:>10} "
              f"{r['cgm']:>8} {r['meals']:>8} {r['boluses']:>8} "
              f"{r['basal_segments']:>8} {r['days']:>6}")


if __name__ == "__main__":
    main()
