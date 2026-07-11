#!/usr/bin/env python3
"""
Parse T1DEXI/OhioT1DM parquet dataset into CSVs for the UVA-Padova pipeline.

Outputs two CSVs:
  {prefix}_cgm_meals_bolus.csv  — CGM trace, meals (mg), boluses (pmol/kg)
  {prefix}_basal.csv            — basal profile segments (U/hr)

Usage:
    python validation/parse_metabonet.py <patient_id> [options]

Examples:
    # All data for patient 559
    python validation/parse_metabonet.py 559

    # First 7 days for patient 1, with weight override
    python validation/parse_metabonet.py 1 --weight 85 --days 7

    # Specific date range for patient 563
    python validation/parse_metabonet.py 563 --date 2021-10-29 --days 3
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta

import pyarrow.compute as pc
import pyarrow.parquet as pq

U_TO_PMOL = 6000.0
DAY_MINUTES = 24 * 60

# Default parquet path — change via --parquet or env UVA_PADOVA_PARQUET
DEFAULT_PARQUET = os.environ.get(
    "UVA_PADOVA_PARQUET",
    r"C:\Users\beleg\Downloads\metabonet_public.parquet",
)


def _resolve_weight(parquet_path: str, patient_id: str, fallback: float | None) -> tuple[float, float] | None:
    """
    Resolve body weight: --weight argument > parquet data.
    Returns (weight_lbs, weight_kg) or None.
    """
    w_lbs = None
    if fallback is not None:
        return (fallback, fallback)

    try:
        t = pq.read_table(
            parquet_path,
            columns=["id", "weight"],
            filters=[("id", "=", patient_id)],
        )
        col = t.column("weight")
        valid = pc.drop_null(col)
        if len(valid) > 0:
            w_lbs = valid[0].as_py()
    except Exception as exc:
        print(f"  [warning] could not read weight from parquet: {exc}")

    if w_lbs is None or w_lbs <= 0:
        return None

    return (w_lbs, w_lbs * 0.453592)


def _aggregate_basal(df):
    """
    Aggregate 5-min basal time series into segments.
    Consecutive rows with the same basal value are merged into one segment
    (keeping the first timestamp).
    """
    if df.empty:
        return []

    segments = []
    prev_rate = df["basal"].iloc[0]
    seg_start = df["date"].iloc[0]

    for idx, row in df.iterrows():
        rate = row["basal"]
        # Treat NaN basal as 0
        if rate is None or (isinstance(rate, float) and rate != rate):
            rate = 0.0
        if abs(rate - prev_rate) > 1e-9:
            segments.append((seg_start, prev_rate))
            seg_start = row["date"]
            prev_rate = rate

    segments.append((seg_start, prev_rate))
    return segments


def minutes_since_ref(dt: datetime, ref: datetime) -> int:
    return int((dt - ref).total_seconds() // 60)


def main():
    ap = argparse.ArgumentParser(description="Parse T1DEXI parquet data to pipeline CSV")
    ap.add_argument("patient_id", help="Patient identifier (e.g. 559, 1, P01)")
    ap.add_argument(
        "--parquet",
        default=DEFAULT_PARQUET,
        help=f"Path to parquet file (default: {DEFAULT_PARQUET})",
    )
    ap.add_argument("--weight", type=float, default=None, help="Body weight in kg")
    ap.add_argument("--date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    ap.add_argument(
        "--days", type=int, default=0, help="Number of days (0 = all)"
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: validation/output)",
    )
    ap.add_argument("--suffix", type=str, default=None, help="Override file suffix")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    patient_id = str(args.patient_id)

    # ---- Resolve weight ----
    weight_res = _resolve_weight(args.parquet, patient_id, args.weight)
    if weight_res is None:
        print(f"Error: no weight found for patient {patient_id} and no --weight given; skipping")
        sys.exit(1)
    weight_lbs, weight = weight_res
    if abs(weight_lbs - weight) < 0.01:
        print(f"Patient: {patient_id}, weight={weight:.1f} kg (from --weight)")
    else:
        print(f"Patient: {patient_id}, weight={weight_lbs:.1f} lbs = {weight:.1f} kg")

    # ---- Read parquet ----
    print("  Reading parquet data...")
    try:
        table = pq.read_table(
            args.parquet,
            columns=["id", "date", "CGM", "basal", "bolus", "carbs", "insulin"],
            filters=[("id", "=", patient_id)],
        )
    except Exception as exc:
        print(f"Error reading parquet: {exc}")
        sys.exit(1)

    if len(table) == 0:
        print(f"Error: no data for patient {patient_id}")
        sys.exit(1)

    print(f"  {len(table)} rows loaded")

    df = table.to_pandas()
    df["date"] = df["date"].dt.tz_localize(None)  # make naive
    df = df.sort_values("date").reset_index(drop=True)

    # ---- Determine time window ----
    first_dt = df["date"].iloc[0]

    if args.date:
        ref_time = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        ref_time = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    if args.days > 0:
        t_end = args.days * DAY_MINUTES
    else:
        max_dt = df["date"].iloc[-1]
        t_end = minutes_since_ref(max_dt, ref_time) + 1
        t_end = ((t_end + DAY_MINUTES - 1) // DAY_MINUTES) * DAY_MINUTES

    print(f"  Window: {ref_time.date()} + {t_end // DAY_MINUTES}d ({t_end} min)")

    # ---- Filter window ----
    df["t_min"] = df["date"].apply(lambda d: minutes_since_ref(d, ref_time))
    df = df[(df["t_min"] >= 0) & (df["t_min"] < t_end)].copy()

    if df.empty:
        print("Error: no data in the specified window")
        sys.exit(1)

    # ---- Build CGM/meals/bolus table ----
    rows = []
    for _, row in df.iterrows():
        cgm = row["CGM"]
        cgm_val = cgm if (cgm is not None and cgm == cgm and cgm > 0) else 0.0

        carbs_g = row["carbs"]
        meal_mg = 0.0
        if carbs_g is not None and carbs_g == carbs_g and carbs_g > 0:
            meal_mg = carbs_g * 1000.0  # g -> mg

        bolus_u = row["bolus"]
        bolus_pmol_kg = 0.0
        if bolus_u is not None and bolus_u == bolus_u and bolus_u > 0:
            bolus_pmol_kg = round(bolus_u * U_TO_PMOL / weight, 4)

        rows.append({
            "t_min": row["t_min"],
            "cgm_mgdl": cgm_val,
            "meal_mg": meal_mg,
            "bolus_pmol_kg": bolus_pmol_kg,
        })

    # ---- Aggregate basal into segments ----
    # Parquet basal is IU delivered in previous 5 min → convert to U/hr
    df_basal = df.copy()
    df_basal["basal"] = (df_basal["basal"] * 12.0).round(6)  # IU/5min -> U/hr
    basal_segments = _aggregate_basal(df_basal)
    basal_rows = []
    for dt, rate in basal_segments:
        t = minutes_since_ref(dt, ref_time)
        if 0 <= t < t_end:
            basal_rows.append({"t_min": t, "basal_u_per_hr": rate})

    # ---- Sort ----
    rows.sort(key=lambda r: r["t_min"])
    basal_rows.sort(key=lambda r: r["t_min"])

    # ---- Determine suffix ----
    if args.suffix:
        suffix = args.suffix
    elif args.days > 0:
        suffix = f"{args.days}d"
    else:
        ndays = t_end // DAY_MINUTES
        suffix = f"{ndays}d"

    # ---- Write CSVs ----
    prefix = f"patient{patient_id}_{suffix}"

    main_path = os.path.join(output_dir, f"{prefix}_cgm_meals_bolus.csv")
    with open(main_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_min", "cgm_mgdl", "meal_mg", "bolus_pmol_kg"])
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows)} rows -> {main_path}")

    basal_path = os.path.join(output_dir, f"{prefix}_basal.csv")
    with open(basal_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_min", "basal_u_per_hr"])
        w.writeheader()
        w.writerows(basal_rows)
    print(f"  {len(basal_rows)} basal segments -> {basal_path}")

    # ---- Summary ----
    n_cgm = sum(1 for r in rows if r["cgm_mgdl"] > 0)
    n_meal = sum(1 for r in rows if r["meal_mg"] > 0)
    n_bolus = sum(1 for r in rows if r["bolus_pmol_kg"] > 0)
    print(f"\nSummary: {n_cgm} CGM, {n_meal} meals, {n_bolus} boluses, "
          f"{len(basal_rows)} basal segments in {t_end} min ({t_end // DAY_MINUTES} days)")


if __name__ == "__main__":
    main()
