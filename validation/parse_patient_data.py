#!/usr/bin/env python3
"""
Parse patient XML (Dexcom + pump logs) into CSV for the UVA-Padova pipeline.

Outputs two CSVs in output_dir/:
  patient{id}_{suffix}_cgm_meals_bolus.csv  — CGM trace, meals (mg), boluses (pmol/kg)
  patient{id}_{suffix}_basal.csv            — basal profile segments (U/hr)

Usage:
    python3 validation/parse_patient_data.py <xml_path> [--date dd-mm-yyyy] [--days N] [--output-dir ...]

Examples:
    # All data (10 days for patient 563)
    python3 validation/parse_patient_data.py <xml_path>

    # Single day (backward compatible)
    python3 validation/parse_patient_data.py <xml_path> --date 29-10-2021

    # First N days starting from --date (or first day if --date omitted)
    python3 validation/parse_patient_data.py <xml_path> --days 2
    python3 validation/parse_patient_data.py <xml_path> --date 29-10-2021 --days 3
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET


U_TO_PMOL = 6000.0


def parse_timestamp(ts_str: str) -> datetime:
    return datetime.strptime(ts_str.strip(), "%d-%m-%Y %H:%M:%S")


def minutes_since_ref(dt: datetime, ref: datetime) -> int:
    return int((dt - ref).total_seconds() // 60)


def extract_patient_info(root: ET.Element) -> tuple:
    pid = root.get("id", "?")
    weight = float(root.get("weight", "0"))
    insulin_type = root.get("insulin_type", "?")
    return pid, weight, insulin_type


def main():
    ap = argparse.ArgumentParser(description="Parse patient XML to pipeline CSV")
    ap.add_argument("xml_path", help="Path to patient XML file")
    ap.add_argument("--date", type=str, default=None,
                    help="Start date (dd-MM-yyyy), e.g. 29-10-2021")
    ap.add_argument("--days", type=int, default=0,
                    help="Number of days to include (0 = all)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: validation/output)")
    ap.add_argument("--suffix", type=str, default=None,
                    help="Override file suffix (default: auto from --days)")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Parse XML ----
    tree = ET.parse(args.xml_path)
    root = tree.getroot()
    pid, weight, insulin_type = extract_patient_info(root)
    print(f"Patient: {pid}, weight={weight} kg, insulin={insulin_type}")

    all_cgm_events = [(parse_timestamp(ev.get("ts")), float(ev.get("value")))
                      for ev in root.findall(".//glucose_level/event")]
    if not all_cgm_events:
        print("Error: no CGM events found")
        sys.exit(1)

    all_meal_events = [(parse_timestamp(ev.get("ts")), float(ev.get("carbs", "0")))
                       for ev in root.findall(".//meal/event")]
    all_bolus_events = [(parse_timestamp(ev.get("ts_begin")), float(ev.get("dose", "0")))
                        for ev in root.findall(".//bolus/event")]
    all_basal_events = [(parse_timestamp(ev.get("ts")), float(ev.get("value", "0")))
                        for ev in root.findall(".//basal/event")]

    # ---- Determine time window ----
    first_cgm_dt = min(dt for dt, _ in all_cgm_events)

    if args.date:
        ref_time = datetime.strptime(args.date, "%d-%m-%Y")
    else:
        # Round first CGM down to midnight
        ref_time = first_cgm_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    day_minutes = 24 * 60

    if args.days > 0:
        t_end = args.days * day_minutes
    else:
        # All data: compute total span from ref_time
        all_dts = [dt for dt, _ in all_cgm_events] + \
                  [dt for dt, _ in all_meal_events] + \
                  [dt for dt, _ in all_bolus_events] + \
                  [dt for dt, _ in all_basal_events]
        max_dt = max(all_dts)
        t_end = minutes_since_ref(max_dt, ref_time) + 1  # +1 for safety
        # Round up to nearest day
        t_end = ((t_end + day_minutes - 1) // day_minutes) * day_minutes

    # ---- Filter events in [0, t_end) ----
    def in_window(dt):
        t = minutes_since_ref(dt, ref_time)
        return 0 <= t < t_end

    cgm_pairs = [(minutes_since_ref(dt, ref_time), v)
                 for dt, v in all_cgm_events if in_window(dt)]
    meal_pairs = [(minutes_since_ref(dt, ref_time), v * 1000.0)  # g -> mg
                  for dt, v in all_meal_events if in_window(dt)]
    bolus_pairs = [(minutes_since_ref(dt, ref_time), v * U_TO_PMOL / weight)
                   for dt, v in all_bolus_events if in_window(dt)]
    basal_pairs = [(minutes_since_ref(dt, ref_time), v)
                   for dt, v in all_basal_events if in_window(dt)]
    basal_pairs.sort(key=lambda x: x[0])

    # Basal profile repeats daily.  If XML only lists 1–2 days, extend.
    if basal_pairs:
        # Detect daily pattern: offets within first 24h
        first24 = [(t, r) for t, r in basal_pairs if t < day_minutes]
        # Check if we need to extend
        last_t = basal_pairs[-1][0]
        n_days_needed = (t_end + day_minutes - 1) // day_minutes
        if first24 and last_t < t_end:
            # Only extend if using the --days flag and we have fewer days than needed
            events_in_window = [(t, r) for t, r in basal_pairs if t < t_end]
            covered_days = (max(t for t, _ in events_in_window) // day_minutes) + 1 if events_in_window else 0
            if covered_days < n_days_needed and args.days > 0:
                # Repeat the first-day pattern
                pattern = list(first24)
                for day in range(covered_days, n_days_needed):
                    for offset, rate in pattern:
                        t = day * day_minutes + offset
                        if t < t_end:
                            basal_pairs.append((t, rate))
                basal_pairs.sort(key=lambda x: x[0])

    basal_pairs = [(t, r) for t, r in basal_pairs if t < t_end]

    if not cgm_pairs:
        print("Error: no CGM events in window")
        sys.exit(1)

    total_min = t_end

    # ---- Build unified table ----
    cgm_times, cgm_values = zip(*cgm_pairs) if cgm_pairs else ([], [])
    meal_dict = dict(meal_pairs)
    bolus_dict = dict(bolus_pairs)

    rows = []
    for t, g in zip(cgm_times, cgm_values):
        rows.append({
            "t_min": t,
            "cgm_mgdl": g,
            "meal_mg": meal_dict.pop(t, 0.0),
            "bolus_pmol_kg": bolus_dict.pop(t, 0.0),
        })

    for t, v in sorted(meal_dict.items()):
        rows.append({"t_min": t, "cgm_mgdl": 0.0, "meal_mg": v, "bolus_pmol_kg": 0.0})
    for t, v in sorted(bolus_dict.items()):
        rows.append({"t_min": t, "cgm_mgdl": 0.0, "meal_mg": 0.0, "bolus_pmol_kg": v})

    rows.sort(key=lambda r: r["t_min"])

    basal_rows = [{"t_min": t, "basal_u_per_hr": r} for t, r in basal_pairs]
    basal_rows.sort(key=lambda r: r["t_min"])

    # ---- Determine suffix ----
    if args.suffix:
        suffix = args.suffix
    elif args.days > 0:
        suffix = f"{args.days}d"
    elif args.date:
        suffix = "1d"  # backward compat
    else:
        ndays = t_end // day_minutes
        suffix = f"{ndays}d"

    # ---- Write CSVs ----
    prefix = f"patient{pid}_{suffix}"

    main_path = os.path.join(output_dir, f"{prefix}_cgm_meals_bolus.csv")
    with open(main_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_min", "cgm_mgdl", "meal_mg", "bolus_pmol_kg"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  {len(rows)} rows -> {main_path}")

    basal_path = os.path.join(output_dir, f"{prefix}_basal.csv")
    with open(basal_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_min", "basal_u_per_hr"])
        w.writeheader()
        for row in basal_rows:
            w.writerow(row)
    print(f"  {len(basal_rows)} basal segments -> {basal_path}")

    n_cgm = sum(1 for r in rows if r["cgm_mgdl"] > 0)
    n_meal = sum(1 for r in rows if r["meal_mg"] > 0)
    n_bolus = sum(1 for r in rows if r["bolus_pmol_kg"] > 0)
    print(f"\nSummary: {n_cgm} CGM, {n_meal} meals, {n_bolus} boluses, "
          f"{len(basal_rows)} basal segments in {total_min} min ({total_min // day_minutes} days)")


if __name__ == "__main__":
    main()
