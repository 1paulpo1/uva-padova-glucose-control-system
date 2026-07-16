from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from parsers.base_parser import BaseParser, PatientData
from data_pipeline import FilterConfig, process_patient


# ══════════════════════════════════════════════════════════════════════════════
# RunSpec — one unit of work: a single patient window
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunSpec:
    study: str
    patient_id: str
    weight_lbs: float
    window_start_days: int   # offset from patient's first data day
    window_days: int          # length of this window


# ══════════════════════════════════════════════════════════════════════════════
# StudyScheduler — scans parquet, groups by study, generates sliding windows
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _PatientInfo:
    pid: str
    study: str
    weight_lbs: float
    first_date: datetime
    total_min: int
    meals: int
    boluses: int
    cgm: int

    @property
    def total_days(self) -> int:
        return max(1, self.total_min // 1440)

    @property
    def cgm_density(self) -> float:
        return self.cgm / max(1, self.total_min // 5)


class StudyScheduler:
    """Build a flat queue of RunSpecs from scan results + sliding windows."""

    def __init__(self, parser: BaseParser, source_path: str):
        self._parser = parser
        self._source = source_path

    def _batch_read_ranges(self, pids: list[str]) -> dict[str, tuple[datetime, int]]:
        """Bulk parquet read: first and last date for all given patients."""
        result: dict[str, tuple[datetime, int]] = {}
        try:
            t = pq.read_table(
                self._source,
                columns=["id", "date"],
                filters=[("id", "in", pids)],
            )
            df = t.to_pandas()
            if "date" not in df.columns or df.empty:
                return result
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
            gb = df.groupby("id")["date"]
            for pid, grp in gb:
                dates = grp.dropna()
                if dates.empty:
                    continue
                first = dates.min()
                last = dates.max()
                total_min = int((last - first).total_seconds() // 60)
                result[str(pid)] = (first, total_min)
        except Exception:
            pass
        return result

    def generate(
        self,
        study_filter: str | None = None,
        window_days: int = 7,
        stride: int = 1,
        sort_by: str = "cgm",
        min_meals: int = 0,
        min_boluses: int = 0,
        max_windows_per_patient: int = 0,
        no_cache: bool = False,
        output_dir: str | None = None,
    ) -> list[RunSpec]:
        scan_cache_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
        sig = None if no_cache else BaseParser._parquet_signature(self._source)
        all_pts = (None if no_cache
                   else BaseParser._load_scan_cache(scan_cache_dir, sig, study_filter))

        if all_pts is None:
            all_pts = self._parser.scan(self._source, study_filter)
            if not all_pts:
                return []
            BaseParser._save_scan_cache(scan_cache_dir, all_pts, sig, study_filter)

        # Filter by meals / boluses
        candidates: list[tuple[str, dict]] = []
        for pid, p in all_pts.items():
            if p["meals"] >= min_meals and p["boluses"] >= min_boluses:
                candidates.append((pid, p))

        if not candidates:
            return []

        # Batch-read date ranges for all candidates at once
        candidate_pids = [pid for pid, _ in candidates]
        ranges = self._batch_read_ranges(candidate_pids)

        # Build _PatientInfo per candidate
        infos: list[_PatientInfo] = []
        for pid, p in candidates:
            rng = ranges.get(pid)
            if rng is None:
                continue
            first_date, total_min = rng
            info = _PatientInfo(
                pid=pid,
                study=p["source_file"],
                weight_lbs=p["weight_lbs"],
                first_date=first_date,
                total_min=total_min,
                meals=p["meals"],
                boluses=p["boluses"],
                cgm=p["cgm"],
            )
            infos.append(info)

        # Sort within each study
        sort_key = {"cgm": lambda x: -x.cgm_density,
                     "meals": lambda x: -x.meals,
                     "boluses": lambda x: -x.boluses}.get(sort_by, lambda x: -x.cgm_density)

        by_study: dict[str, list[_PatientInfo]] = defaultdict(list)
        for info in infos:
            by_study[info.study].append(info)

        specs: list[RunSpec] = []
        for study_name in sorted(by_study):
            patients = sorted(by_study[study_name], key=sort_key)
            for info in patients:
                n_windows = 0
                max_start = info.total_days - window_days
                if max_start < 0:
                    max_start = 0
                for offset in range(0, max_start + 1, stride):
                    if max_windows_per_patient > 0 and n_windows >= max_windows_per_patient:
                        break
                    specs.append(RunSpec(
                        study=study_name,
                        patient_id=info.pid,
                        weight_lbs=info.weight_lbs,
                        window_start_days=offset,
                        window_days=window_days,
                    ))
                    n_windows += 1

        return specs


# ══════════════════════════════════════════════════════════════════════════════
# PipelineOrchestrator — executes RunSpecs in progressive batches
# ══════════════════════════════════════════════════════════════════════════════

_MINGW_BIN = r"C:\msys64\mingw64\bin"
_SUBPROCESS_ENV = os.environ.copy()
_SUBPROCESS_ENV["PATH"] = f"{_MINGW_BIN};{_SUBPROCESS_ENV.get('PATH', '')}"


def _make_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_patient(test_real_path: str, cgm_csv: str, basal_csv: str,
                 json_out: str, restarts: int, evals: int,
                 timeout: int = 600, verbose: bool = False,
                 params_config: str | None = None) -> dict:
    cmd = [test_real_path, "--cgm", cgm_csv, "--basal", basal_csv,
           "--json", json_out, "--restarts", str(restarts), "--evals", str(evals)]
    if params_config:
        cmd += ["--params-config", params_config]

    t0 = time.time()
    stdout = ""
    stderr = ""
    try:
        if verbose:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=_SUBPROCESS_ENV)
            lines = []
            for line in iter(proc.stdout.readline, ''):
                lines.append(line)
                print(line, end='', flush=True)
            proc.stdout.close()
            ret = proc.wait(timeout=timeout)
            stderr = proc.stderr.read()
            proc.stderr.close()
            stdout = ''.join(lines)
        else:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                env=_SUBPROCESS_ENV)
            stdout = proc.stdout
            stderr = proc.stderr
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}
    except FileNotFoundError:
        return {"error": f"test_real_data not found at {test_real_path}"}
    except KeyboardInterrupt:
        return {"error": "interrupted"}
    except Exception as exc:
        return {"error": str(exc)}

    wall = time.time() - t0
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
        parsed["stderr"] = stderr[:1000] if stderr else ""
    return parsed


def _evaluate_batch(results_batch: list[dict], thresholds: dict) -> tuple[bool, str | None]:
    n = len(results_batch)
    if n == 0:
        return (False, None)
    n_err = sum(1 for r in results_batch if r.get("error"))
    n_skip = sum(1 for r in results_batch
                 if r.get("error", "").startswith("no CGM")
                 or r.get("error", "").startswith("CGM density"))
    n_crash = n_err - n_skip
    n_ok = n - n_err
    n_gate_a_fail = sum(1 for r in results_batch
                        if not r.get("error") and not r.get("gate_a", {}).get("passed", False))
    n_gate_b_fail = sum(1 for r in results_batch
                        if not r.get("error") and not r.get("gate_b", {}).get("passed", False))
    n_pass = sum(1 for r in results_batch
                 if not r.get("error")
                 and r.get("gate_a", {}).get("passed", False)
                 and r.get("gate_b", {}).get("passed", False))

    crash_denom = n - n_skip
    crash_rate = n_crash / max(1, crash_denom)
    if crash_rate > thresholds.get("crash", 0.5):
        return (True, f"Crash rate {crash_rate:.0%} > {thresholds['crash']:.0%} ({n_crash}/{crash_denom})")
    if n_ok == 0 and n_crash > 0:
        return (True, f"All {n} windows crashed")
    rate_a = n_gate_a_fail / n_ok if n_ok > 0 else 0
    rate_b = n_gate_b_fail / n_ok if n_ok > 0 else 0
    rate_pass = n_pass / n_ok if n_ok > 0 else 0
    if rate_a > thresholds["gate_a"]:
        return (True, f"Gate A failures {rate_a:.0%} > {thresholds['gate_a']:.0%}")
    if rate_b > thresholds["gate_b"]:
        return (True, f"Gate B failures {rate_b:.0%} > {thresholds['gate_b']:.0%}")
    if rate_pass < thresholds["pass_rate_min"]:
        return (True, f"Pass rate {rate_pass:.0%} < {thresholds['pass_rate_min']:.0%}")
    return (False, None)


def _print_dashboard(batch_results: list[dict], batch_idx: int, batch_size: int,
                     thresholds: dict, stop_reason: str | None = None):
    n = len(batch_results)
    if n == 0:
        return
    n_err = sum(1 for r in batch_results if r.get("error"))
    n_skip = sum(1 for r in batch_results
                 if r.get("error", "").startswith("no CGM")
                 or r.get("error", "").startswith("CGM density"))
    n_crash = n_err - n_skip
    n_ok = n - n_err
    n_gate_a = sum(1 for r in batch_results
                   if not r.get("error") and r.get("gate_a", {}).get("passed", False))
    n_gate_b = sum(1 for r in batch_results
                   if not r.get("error") and r.get("gate_b", {}).get("passed", False))
    n_pass = sum(1 for r in batch_results
                 if not r.get("error")
                 and r.get("gate_a", {}).get("passed", False)
                 and r.get("gate_b", {}).get("passed", False))

    conds = [r.get("gate_c", {}).get("condition_number", 0)
             for r in batch_results if not r.get("error") and "gate_c" in r]
    conds = [c for c in conds if c > 0]
    costs = [r.get("cost", 0) for r in batch_results if not r.get("error") and r.get("cost")]

    print()
    print(f"{'=' * 62}")
    print(f"  Batch {batch_idx} | N={n} (requested {batch_size})")
    print(f"{'=' * 62}")
    ids_str = " ".join(str(r.get("id", "?")) for r in batch_results)
    print(f"  windows: {ids_str}")
    print(f"  completed: {n}/{n}")
    print()
    if n_ok > 0:
        print(f"  Gate A pass: {n_gate_a}/{n_ok} ({n_gate_a/n_ok:.0%})")
        print(f"  Gate B pass: {n_gate_b}/{n_ok} ({n_gate_b/n_ok:.0%})")
        print(f"  Overall:     {n_pass}/{n_ok} ({n_pass/n_ok:.0%})")
    else:
        print(f"  All {n_err} windows failed")
    print(f"  Skipped:     {n_skip}")
    print(f"  Crashes:     {n_crash}")
    if conds:
        print(f"  Condition #: min={min(conds):.1e}  median={sorted(conds)[len(conds)//2]:.1e}  max={max(conds):.1e}")
    if costs:
        print(f"  Cost:        min={min(costs):.3e}  median={sorted(costs)[len(costs)//2]:.3e}  max={max(costs):.3e}")
    avg_time = sum(r.get("wall_time_s", 0) for r in batch_results) / n if n > 0 else 0
    print(f"  Avg wall:    {avg_time:.1f}s/window")
    print()
    if stop_reason:
        print(f"  !! EARLY STOP: {stop_reason}")
    else:
        next_n = n * 2
        print(f"  -> scaling up to {next_n}")
    print(f"{'=' * 62}")
    print()


class PipelineOrchestrator:
    """Progressive batch execution of RunSpecs against test_real_data.exe."""

    def __init__(
        self,
        parser: BaseParser,
        source_path: str,
        test_real_data_path: str,
        output_dir: str,
        filter_config: Optional[FilterConfig] = None,
        params_config: Optional[str] = None,
        restarts: int = 3,
        evals: int = 500,
        timeout: int = 600,
        verbose: bool = False,
        min_cgm_density: float = 0.5,
    ):
        self._parser = parser
        self._source = source_path
        self._test_real = test_real_data_path
        self._output = output_dir
        self._filter_config = filter_config or FilterConfig()
        self._params_config = params_config
        self._restarts = restarts
        self._evals = evals
        self._timeout = timeout
        self._verbose = verbose
        self._min_cgm_density = min_cgm_density

    def _suffix(self, spec: RunSpec) -> str:
        return f"o{spec.window_start_days}_{spec.window_days}d"

    def _process_one(self, spec: RunSpec, gb_store: dict[str, float]) -> dict | None:
        """Run one window. Returns result dict, or None if filter-rejected (caller should retry)."""
        pid = spec.patient_id
        suffix = self._suffix(spec)
        cgm_csv = os.path.join(self._output, f"patient{pid}_{suffix}_cgm_meals_bolus.csv")
        basal_csv = os.path.join(self._output, f"patient{pid}_{suffix}_basal.csv")

        # Parse with offset
        first_date = self._get_first_date(pid)  # cached internally
        if first_date is None:
            return {"id": pid, "error": "no_date_range", "spec": spec}
        start_str = (first_date + timedelta(days=spec.window_start_days)).strftime("%Y-%m-%d")

        try:
            data = self._parser.parse_patient(
                pid, self._source, days=spec.window_days, start_date=start_str)
        except ValueError as e:
            return {"id": pid, "error": f"csv_gen: {e}", "spec": spec}

        # Filter
        data, meta = process_patient(data, self._filter_config)
        for w in meta.warnings:
            print(f"  [{pid}] warning: {w}")

        if not meta.valid:
            return {"id": pid, "error": f"filter: {'; '.join(meta.warnings[:3])}",
                    "gb": meta.gb_estimate, "cgm_density": meta.cgm_density, "spec": spec}

        gb_store[pid] = meta.gb_estimate

        # Write CSVs
        BaseParser.write_csvs(data, self._output, suffix=suffix)

        # Check CSV quality
        quality = BaseParser.check_csv_quality(cgm_csv, basal_csv)
        if not quality["has_data"]:
            self._cleanup(cgm_csv, basal_csv)
            return {"id": pid, "error": "no CGM data in CSV", "spec": spec}
        cgm_density = quality.get("cgm_density", 0)
        if self._min_cgm_density > 0 and cgm_density < self._min_cgm_density:
            self._cleanup(cgm_csv, basal_csv)
            return {"id": pid, "error": f"CGM density {cgm_density:.0%}", "spec": spec}

        # Run test_real_data
        json_out = os.path.join(self._output, f"patient{pid}_{suffix}_result.json")
        if self._verbose:
            print(f"-- [{pid}] test_real_data --")
        else:
            print(f"  [{pid}] running test_real_data...", end=" ", flush=True)
        tout = self._timeout if self._timeout > 0 else None
        result = _run_patient(
            self._test_real, cgm_csv, basal_csv, json_out,
            self._restarts, self._evals, tout,
            verbose=self._verbose, params_config=self._params_config)
        result["id"] = pid
        result["spec"] = spec
        result["cgm_density"] = cgm_density
        if pid in gb_store:
            result["gb"] = gb_store[pid]

        if result.get("error"):
            print(f"ERROR ({result['error']})")
        else:
            ga = result.get("gate_a", {}).get("passed", False)
            gb = result.get("gate_b", {}).get("passed", False)
            flags = []
            flags.append("GA+" if ga else "GA-")
            flags.append("GB+" if gb else "GB-")
            cn = result.get("gate_c", {}).get("condition_number", 0)
            if cn:
                flags.append(f"cond={cn:.1e}")
            print(f"{' '.join(flags)}  ({result.get('wall_time_s', 0):.1f}s)")

        return result

    def _get_first_date(self, pid: str) -> datetime | None:
        if not hasattr(self, "_first_date_cache"):
            self._first_date_cache: dict[str, datetime] = {}
        if pid not in self._first_date_cache:
            try:
                t = pq.read_table(
                    self._source, columns=["id", "date"],
                    filters=[("id", "=", pid)])
                dates = [d for d in t.column("date").to_pylist() if d is not None]
                if not dates:
                    return None
                d = min(dates)
                if hasattr(d, "tzinfo") and d.tzinfo is not None:
                    d = d.replace(tzinfo=None)
                self._first_date_cache[pid] = d
            except Exception:
                return None
        return self._first_date_cache[pid]

    @staticmethod
    def _cleanup(*paths: str):
        for p in paths:
            if os.path.exists(p):
                os.remove(p)

    def run(
        self,
        specs: list[RunSpec],
        batch_start: int = 5,
        batch_multiplier: float = 2.0,
        max_patients: int = 200,
        results_path: str | None = None,
        thresholds: dict | None = None,
    ) -> list[dict]:
        if thresholds is None:
            thresholds = {"gate_a": 0.80, "gate_b": 0.80, "pass_rate_min": 0.20, "crash": 0.5}

        # Group specs by patient for whole-patient batching
        by_patient: dict[str, list[RunSpec]] = defaultdict(list)
        for sp in specs:
            by_patient[sp.patient_id].append(sp)

        # Sort patients by their first spec's study, then quality
        patient_queue: list[tuple[str, list[RunSpec]]] = list(by_patient.items())

        gb_store: dict[str, float] = {}
        batch_idx = 1
        batch_size = batch_start
        total_done = 0
        stop_reason: str | None = None
        all_results: list[dict] = []
        processed_patients: set[str] = set()

        while total_done < max_patients and patient_queue:
            # Take a batch of unique patients
            batch_patients = patient_queue[:batch_size]
            patient_queue = patient_queue[batch_size:]

            # Skip already-processed patients
            batch_patients = [(pid, wins) for pid, wins in batch_patients
                              if pid not in processed_patients]
            if not batch_patients:
                batch_size = int(batch_size * batch_multiplier)
                batch_idx += 1
                continue

            batch_results: list[dict] = []
            batch_record = {
                "batch": batch_idx,
                "batch_size": batch_size,
                "n_patients": len(batch_patients),
                "start_time": _make_timestamp(),
                "patients": [],
            }

            print(f"\nBatch {batch_idx}: {len(batch_patients)} patients"
                  f" ({total_done}/{max_patients} done)")

            for pid, windows in batch_patients:
                processed_patients.add(pid)
                total_done += 1

                patient_window_results: list[dict] = []
                for spec in windows:
                    result = self._process_one(spec, gb_store)
                    if result is None:
                        continue
                    patient_window_results.append(result)

                    if result.get("error") == "interrupted":
                        stop_reason = "interrupted"
                        break
                batch_results.extend(patient_window_results)
                all_results.extend(patient_window_results)

                # Save incremental results (all windows of this patient)
                if results_path and patient_window_results:
                    self._save_incremental(results_path, patient_window_results,
                                           sorted(processed_patients))

                if stop_reason:
                    break

            if stop_reason:
                break

            batch_record["end_time"] = _make_timestamp()
            batch_record["patients"] = batch_results

            stop, stop_reason = _evaluate_batch(batch_results, thresholds)
            _print_dashboard(batch_results, batch_idx, batch_size, thresholds, stop_reason)

            batch_idx += 1
            if stop:
                break

            batch_size = int(batch_size * batch_multiplier)
            if batch_size > len(patient_queue):
                batch_size = len(patient_queue)

        return all_results

    @staticmethod
    def _save_incremental(results_path: str, new_results: list[dict], processed_ids: list):
        existing = {"batches": [], "processed_ids": []}
        if os.path.exists(results_path):
            try:
                with open(results_path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        for r in new_results:
            existing["batches"].append({"patient": r})
        existing["processed_ids"] = processed_ids
        tmp = results_path + ".tmp"
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        os.replace(tmp, results_path)
