#!/usr/bin/env python3
"""
Staged validation pipeline for T1DEXI/OhioT1DM dataset.

Design:
  Phase 0: scan parquet → filter patients → generate CSVs
  Phase 1: run test_real_data (mode=config) on progressive batches
  Early-stop if gates fail systematically.

Usage:
    python validation/run_validation.py
    python validation/run_validation.py --study Loop --batch-start 5 --max-patients 50
    python validation/run_validation.py --resume validation_results.json
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from parsers import discover, get_parser_for_source, BaseParser

# MinGW-w64 DLL path — needed for test_real_data.exe
_MINGW_BIN = r"C:\msys64\mingw64\bin"
_SUBPROCESS_ENV = os.environ.copy()
_SUBPROCESS_ENV["PATH"] = f"{_MINGW_BIN};{_SUBPROCESS_ENV.get('PATH', '')}"

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PARQUET = os.environ.get(
    "UVA_PADOVA_PARQUET",
    r"C:\Users\beleg\Downloads\metabonet_public.parquet",
)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — run test_real_data.exe and read JSON output
# ──────────────────────────────────────────────────────────────────────────────
def run_patient(test_real_path: str, cgm_csv: str, basal_csv: str,
                json_out: str, restarts: int, evals: int,
                timeout: int = 600) -> dict:
    """Run test_real_data.exe on one patient and return parsed JSON result."""
    cmd = [test_real_path,
           "--cgm", cgm_csv,
           "--basal", basal_csv,
           "--json", json_out,
           "--restarts", str(restarts),
           "--evals", str(evals)]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env=_SUBPROCESS_ENV)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}
    except FileNotFoundError:
        return {"error": f"test_real_data not found at {test_real_path}"}
    except KeyboardInterrupt:
        return {"error": "interrupted"}
    except Exception as exc:
        return {"error": str(exc)}

    wall = time.time() - t0

    # Read JSON output
    if os.path.exists(json_out):
        try:
            with open(json_out) as f:
                parsed = json.load(f)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            return {"error": f"invalid JSON output: {exc}"}
    else:
        return {"error": f"JSON output not found at {json_out}"}

    parsed["exit_code"] = proc.returncode
    parsed["wall_time_s"] = round(wall, 2)

    if proc.returncode != 0:
        parsed["error"] = parsed.get("error") or f"exit code {proc.returncode}"
        parsed["stderr"] = proc.stderr[:1000] if proc.stderr else ""

    return parsed


# ──────────────────────────────────────────────────────────────────────────────
# Batch logic
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_batch(patients_batch: list, thresholds: dict) -> tuple[bool, str | None]:
    """Check early-stopping conditions. Returns (stop, reason)."""
    n = len(patients_batch)
    if n == 0:
        return (False, None)

    n_err = sum(1 for p in patients_batch if p.get("error"))
    # SKIP (no CGM data) is not a crash — it's a data quality filter reject
    n_skip = sum(1 for p in patients_batch
                 if p.get("error", "").startswith("no CGM")
                 or p.get("error", "").startswith("CGM density"))
    n_crash = n_err - n_skip
    n_ok = n - n_err
    n_gate_a_fail = sum(1 for p in patients_batch
                        if not p.get("error") and not p.get("gate_a", {}).get("passed", False))
    n_gate_b_fail = sum(1 for p in patients_batch
                        if not p.get("error") and not p.get("gate_b", {}).get("passed", False))
    n_pass = sum(1 for p in patients_batch
                 if not p.get("error")
                 and p.get("gate_a", {}).get("passed", False)
                 and p.get("gate_b", {}).get("passed", False))

    # Crash rate only counts real crashes (not SKIPs)
    crash_denom = n - n_skip
    crash_rate = n_crash / max(1, crash_denom)
    if crash_rate > thresholds.get("crash", 0.5):
        return (True, f"Crash rate {crash_rate:.0%} > {thresholds['crash']:.0%} ({n_crash}/{crash_denom})")
    if n_ok == 0 and n_crash > 0:
        return (True, f"All {n} patients crashed")

    rate_a = n_gate_a_fail / n_ok
    rate_b = n_gate_b_fail / n_ok
    rate_pass = n_pass / n_ok

    if rate_a > thresholds["gate_a"]:
        return (True, f"Gate A failures {rate_a:.0%} > {thresholds['gate_a']:.0%}")
    if rate_b > thresholds["gate_b"]:
        return (True, f"Gate B failures {rate_b:.0%} > {thresholds['gate_b']:.0%}")
    if rate_pass < thresholds["pass_rate_min"]:
        return (True, f"Pass rate {rate_pass:.0%} < {thresholds['pass_rate_min']:.0%}")

    return (False, None)


def _make_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ──────────────────────────────────────────────────────────────────────────────
# Results persistence
# ──────────────────────────────────────────────────────────────────────────────
def load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"batches": [], "meta": {}, "processed_ids": []}


def save_results(path: str, results: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────────────────
def print_dashboard(patients_batch: list, batch_idx: int, batch_size: int,
                    thresholds: dict, stop_reason: str | None = None):
    n = len(patients_batch)
    if n == 0:
        return
    n_err = sum(1 for p in patients_batch if p.get("error"))
    n_skip = sum(1 for p in patients_batch
                 if p.get("error", "").startswith("no CGM")
                 or p.get("error", "").startswith("CGM density"))
    n_crash = n_err - n_skip
    n_ok = n - n_err
    n_gate_a = sum(1 for p in patients_batch
                   if not p.get("error") and p.get("gate_a", {}).get("passed", False))
    n_gate_b = sum(1 for p in patients_batch
                   if not p.get("error") and p.get("gate_b", {}).get("passed", False))
    n_pass = sum(1 for p in patients_batch
                 if not p.get("error")
                 and p.get("gate_a", {}).get("passed", False)
                 and p.get("gate_b", {}).get("passed", False))

    conds = [p.get("gate_c", {}).get("condition_number", 0)
             for p in patients_batch if not p.get("error") and "gate_c" in p]
    conds = [c for c in conds if c > 0]
    costs = [p.get("cost", 0) for p in patients_batch
             if not p.get("error") and p.get("cost")]

    print()
    print(f"{'=' * 62}")
    print(f"  Batch {batch_idx} | N={n} (requested {batch_size})")
    print(f"{'=' * 62}")
    ids_str = " ".join(str(p.get("id", "?")) for p in patients_batch)
    print(f"  patients: {ids_str}")
    print(f"  completed: {n}/{n}")
    print()
    if n_ok > 0:
        print(f"  Gate A pass: {n_gate_a}/{n_ok} ({n_gate_a/n_ok:.0%})"
              f"{'  x threshold ' + str(thresholds['gate_a']*100) + '%' if n_gate_a/n_ok <= 1-thresholds['gate_a'] else ''}")
        print(f"  Gate B pass: {n_gate_b}/{n_ok} ({n_gate_b/n_ok:.0%})"
              f"{'  x threshold ' + str(thresholds['gate_b']*100) + '%' if n_gate_b/n_ok <= 1-thresholds['gate_b'] else ''}")
        print(f"  Overall:     {n_pass}/{n_ok} ({n_pass/n_ok:.0%})"
              f"{'  x min ' + str(thresholds['pass_rate_min']*100) + '%' if n_pass/n_ok < thresholds['pass_rate_min'] else ''}")
    else:
        print(f"  All {n_err} patients failed -- no pipeline results")
    print(f"  Skipped:     {n_skip}")
    print(f"  Crashes:     {n_crash}")
    if conds:
        print(f"  Condition #: min={min(conds):.1e}  median={sorted(conds)[len(conds)//2]:.1e}  max={max(conds):.1e}")
    if costs:
        print(f"  Cost:        min={min(costs):.3e}  median={sorted(costs)[len(costs)//2]:.3e}  max={max(costs):.3e}")
    densities = [p.get("cgm_density", 0) for p in patients_batch
                 if p.get("cgm_density", 0) > 0]
    if densities:
        d_min, d_avg, d_max = min(densities), sum(densities)/len(densities), max(densities)
        print(f"  CGM density: min={d_min:.0%}  avg={d_avg:.0%}  max={d_max:.0%}")
    avg_time = sum(p.get("wall_time_s", 0) for p in patients_batch) / n
    print(f"  Avg wall:    {avg_time:.1f}s/patient")
    print()

    if stop_reason:
        print(f"  !! EARLY STOP: {stop_reason}")
    else:
        next_n = n * 2
        print(f"  -> pass rate OK, scaling up to {next_n}")
    print(f"{'=' * 62}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Staged validation on T1DEXI / OhioT1DM")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--study", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--test-real-data", default=None,
                    help="Path to test_real_data.exe")
    ap.add_argument("--resume", default=None,
                    help="Path to existing results JSON")
    ap.add_argument("--batch-start", type=int, default=5)
    ap.add_argument("--batch-multiplier", type=float, default=2.0)
    ap.add_argument("--max-patients", type=int, default=200)
    ap.add_argument("--min-meals", type=int, default=5)
    ap.add_argument("--min-boluses", type=int, default=5)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--gate-a-threshold", type=float, default=0.80)
    ap.add_argument("--gate-b-threshold", type=float, default=0.80)
    ap.add_argument("--pass-rate-min", type=float, default=0.20)
    ap.add_argument("--cond-max", type=float, default=1e6,
                    help="Not a blocker, only reported")
    ap.add_argument("--no-cache", action="store_true",
                    help="Force re-scan parquet, ignore cache")
    ap.add_argument("--crash-threshold", type=float, default=0.5,
                    help="Max crash fraction before early-stop (default: 0.5)")
    ap.add_argument("--min-cgm-density", type=float, default=0.5,
                    help="Min CGM density (non-zero / expected 5-min slots, default: 0.5)")
    args = ap.parse_args()

    # ── Init parser ──────────────────────────────────────────────────────
    discover()
    parser_cls = get_parser_for_source(args.parquet)
    if parser_cls is None:
        print(f"Error: no parser found for {args.parquet}")
        sys.exit(1)
    parser = parser_cls()

    # ── Resolve paths ──────────────────────────────────────────────────
    script_dir = Path(__file__).parent.resolve()
    output_dir = Path(args.output_dir or script_dir / "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_real_path = args.test_real_data or (
            script_dir.parent / "build" / "validation" / "test_real_data.exe")
    test_real_path = Path(test_real_path)
    if not test_real_path.exists():
        print(f"Error: test_real_data not found at {test_real_path}")
        sys.exit(1)

    results_path = output_dir / "validation_results.json"
    csv_summary_path = output_dir / "validation_results.csv"

    thresholds = {
        "gate_a": args.gate_a_threshold,
        "gate_b": args.gate_b_threshold,
        "pass_rate_min": args.pass_rate_min,
        "crash": args.crash_threshold,
    }

    # ── Load or init results ───────────────────────────────────────────
    if args.resume:
        results = load_results(args.resume)
        print(f"Resumed from {args.resume}: {len(results['processed_ids'])} patients done")
        processed = set(results["processed_ids"])
    else:
        results = {"batches": [], "meta": {}, "processed_ids": []}
        processed = set()

    results.setdefault("meta", {})
    results["meta"].update({
        "study": args.study or "all",
        "restarts": args.restarts,
        "evals": args.evals,
        "days": args.days,
        "thresholds": thresholds,
        "started_at": results["meta"].get("started_at") or _make_timestamp(),
    })

    # ── Scan patients (with cache) ─────────────────────────────────────
    print("Phase 0: scanning parquet...")
    sig = None if args.no_cache else BaseParser._parquet_signature(args.parquet)
    all_pts = (None if args.no_cache
               else BaseParser._load_scan_cache(str(output_dir), sig, args.study))

    if all_pts is None:
        all_pts = parser.scan(args.parquet, args.study)
        if not all_pts:
            print("No patients found!")
            sys.exit(1)
        BaseParser._save_scan_cache(str(output_dir), all_pts, sig, args.study)
        print(f"  Found {len(all_pts)} patients with weight")
    else:
        print(f"  Cache hit: {len(all_pts)} patients with weight")

    # Filter by meals/boluses
    candidates = []
    for pid, p in all_pts.items():
        if p["meals"] >= args.min_meals and p["boluses"] >= args.min_boluses:
            candidates.append((pid, p["weight_lbs"]))

    # Window check
    print(f"  Checking N-day windows...")
    eligible = []
    min_cgm_points = max(1, int(args.days * 288 * args.min_cgm_density))
    for pid, w_lbs in candidates:
        passes, meals_w, boluses_w, cgm_w = parser.check_window(
            args.parquet, pid, args.days, args.min_meals, args.min_boluses, min_cgm_points)
        if passes:
            eligible.append((pid, w_lbs))
    print(f"  Eligible (meals>={args.min_meals}, bolus>={args.min_boluses}, "
          f"cgm>={min_cgm_points} in {args.days}d): {len(eligible)} patients")

    # Shuffle for diversity
    import random
    random.seed(42)
    random.shuffle(eligible)

    # Unlimited queue
    patient_queue = eligible[:]
    print(f"  Queue ready: {len(patient_queue)} patients")

    # ── Validation loop ────────────────────────────────────────────────
    batch_size = args.batch_start
    batch_idx = len(results["batches"]) + 1
    total_done = len(processed)
    stop_reason = None

    while total_done < args.max_patients and len(patient_queue) > 0:
        # Take next batch
        batch_patients = patient_queue[:batch_size]
        patient_queue = patient_queue[batch_size:]

        # Skip already processed
        batch_patients = [(pid, w) for pid, w in batch_patients
                          if pid not in processed]
        if not batch_patients:
            batch_size = int(batch_size * args.batch_multiplier)
            batch_idx += 1
            continue

        batch_record = {
            "batch": batch_idx,
            "batch_size": batch_size,
            "n_patients": len(batch_patients),
            "start_time": _make_timestamp(),
            "patients": [],
        }

        print(f"\nBatch {batch_idx}: {len(batch_patients)} patients"
              f" ({total_done}/{args.max_patients} done)")

        for pid, weight_lbs in batch_patients:
            suffix = f"{args.days}d"
            cgm_csv = output_dir / f"patient{pid}_{suffix}_cgm_meals_bolus.csv"
            basal_csv = output_dir / f"patient{pid}_{suffix}_basal.csv"

            # Generate CSV if needed (in-process, no subprocess)
            if not cgm_csv.exists():
                print(f"  [{pid}] generating CSV...", end=" ", flush=True)
                try:
                    data = parser.parse_patient(pid, args.parquet, days=args.days)
                    cgm_csv, basal_csv = BaseParser.write_csvs(data, str(output_dir))
                except ValueError as e:
                    print(f"SKIP ({e})")
                    pr = {"id": pid, "batch": batch_idx, "error": f"csv_gen: {e}"}
                    batch_record["patients"].append(pr)
                    processed.add(pid)
                    total_done += 1
                    results["batches"].append({"batch": batch_idx, "patient": pr})
                    results["processed_ids"] = sorted(processed)
                    save_results(str(results_path), results)
                    continue
                print("OK")

            # Check CSV quality
            quality = BaseParser.check_csv_quality(str(cgm_csv), str(basal_csv))
            cgm_density = quality.get("cgm_density", 0)
            if not quality["has_data"]:
                print(f"  [{pid}] SKIP (no CGM data)")
                pr = {"id": pid, "batch": batch_idx,
                      "error": "no CGM data in CSV"}
                batch_record["patients"].append(pr)
                processed.add(pid)
                total_done += 1
                results["batches"].append({"batch": batch_idx, "patient": pr})
                results["processed_ids"] = sorted(processed)
                save_results(str(results_path), results)
                for p in [cgm_csv, basal_csv]:
                    if p.exists():
                        p.unlink()
                continue
            if args.min_cgm_density > 0 and cgm_density < args.min_cgm_density:
                print(f"  [{pid}] SKIP (CGM density {cgm_density:.0%} < {args.min_cgm_density:.0%})")
                pr = {"id": pid, "batch": batch_idx,
                      "error": f"CGM density {cgm_density:.0%}"}
                batch_record["patients"].append(pr)
                processed.add(pid)
                total_done += 1
                results["batches"].append({"batch": batch_idx, "patient": pr})
                results["processed_ids"] = sorted(processed)
                save_results(str(results_path), results)
                for p in [cgm_csv, basal_csv]:
                    if p.exists():
                        p.unlink()
                continue

            # Run test_real_data
            json_out = output_dir / f"patient{pid}_{suffix}_result.json"
            print(f"  [{pid}] running test_real_data...", end=" ", flush=True)
            pr = run_patient(str(test_real_path), str(cgm_csv), str(basal_csv),
                             str(json_out), args.restarts, args.evals, args.timeout)
            pr["id"] = pid
            pr["batch"] = batch_idx
            pr["cgm_density"] = cgm_density
            batch_record["patients"].append(pr)

            if pr.get("error"):
                print(f"ERROR ({pr['error']})")
                if pr["error"] == "interrupted":
                    stop_reason = "interrupted"
                    break
            else:
                ga = pr.get("gate_a", {}).get("passed", False)
                gb = pr.get("gate_b", {}).get("passed", False)
                flags = []
                flags.append("GA+" if ga else "GA-")
                flags.append("GB+" if gb else "GB-")
                cn = pr.get("gate_c", {}).get("condition_number", 0)
                if cn:
                    flags.append(f"cond={cn:.1e}")
                print(f"{' '.join(flags)}  ({pr.get('wall_time_s',0):.1f}s)")

            processed.add(pid)
            total_done += 1

            # Append per-patient to results file
            results["batches"].append({
                "batch": batch_idx,
                "patient": pr,
            })
            results["processed_ids"] = sorted(processed)
            save_results(str(results_path), results)

        # ── Evaluate batch ────────────────────────────────────────────
        batch_record["end_time"] = _make_timestamp()

        stop, stop_reason = evaluate_batch(
            batch_record["patients"], thresholds)
        stop_reason = stop_reason or ""

        print_dashboard(batch_record["patients"], batch_idx, batch_size,
                        thresholds, stop_reason)

        batch_idx += 1

        if stop:
            break

        # Scale up
        batch_size = int(batch_size * args.batch_multiplier)
        if batch_size > len(patient_queue):
            batch_size = len(patient_queue)

    # ── Final summary ──────────────────────────────────────────────────
    results["meta"]["completed_at"] = _make_timestamp()
    results["meta"]["total_processed"] = total_done
    results["meta"]["stopped_at"] = {
        "reason": stop_reason or "completed",
        "batch": batch_idx - 1,
    }
    save_results(str(results_path), results)

    # Write CSV summary
    with open(csv_summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "patient", "error", "cost", "success",
                     "gate_a", "gate_b", "cond_number", "wall_time_s"])
        for entry in results["batches"]:
            pr = entry["patient"]
            w.writerow([
                entry["batch"],
                pr.get("id"),
                pr.get("error", ""),
                pr.get("cost", ""),
                pr.get("success", ""),
                pr.get("gate_a", {}).get("passed", ""),
                pr.get("gate_b", {}).get("passed", ""),
                pr.get("gate_c", {}).get("condition_number", ""),
                pr.get("wall_time_s", ""),
            ])

    # ── Build analysis envelope from per-patient JSONs ──────────────────
    analysis_results_path = output_dir / "validation_results_analysis.json"
    param_names = None
    envelope_results = []
    suffix = f"{args.days}d"
    for entry in results["batches"]:
        pr = entry["patient"]
        pid = pr.get("id")
        if pid is None:
            continue
        json_path = output_dir / f"patient{pid}_{suffix}_result.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    pat = json.load(f)
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                print(f"  WARN: skipped patient {pid} in envelope (corrupt JSON: {exc})")
                continue
            pat["patient_id"] = pid
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

    # Print final stats
    all_pass = sum(1 for e in results["batches"]
                   if not e["patient"].get("error")
                   and e["patient"].get("gate_a", {}).get("passed", False)
                   and e["patient"].get("gate_b", {}).get("passed", False))
    all_total = len(results["batches"])
    print()
    print("=" * 62)
    print("  VALIDATION COMPLETE")
    print("=" * 62)
    print(f"  Patients processed: {all_total}")
    print(f"  All gates passed:   {all_pass} ({all_pass/all_total:.0%}" if all_total > 0 else "  All gates passed:   0")
    print(f"  Batch stop reason:  {stop_reason or 'none (max reached)'}")
    print(f"  Results:            {results_path}")
    print(f"  CSV summary:        {csv_summary_path}")
    print(f"  Analysis envelope:  {analysis_results_path} ({len(envelope_results)} patients)")
    print()


if __name__ == "__main__":
    main()
