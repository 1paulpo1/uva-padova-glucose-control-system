from __future__ import annotations

import csv
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

U_TO_PMOL = 6000.0
DAY_MINUTES = 24 * 60
SCAN_CACHE_FILE = "patient_scan_cache.json"


@dataclass
class PatientData:
    patient_id: str
    weight_kg: float
    source: str
    total_min: int
    rows: list[dict] = field(default_factory=list)
    basal_segments: list[dict] = field(default_factory=list)

    @property
    def n_cgm(self) -> int:
        return sum(1 for r in self.rows if r.get("cgm_mgdl", 0) > 0)

    @property
    def n_meals(self) -> int:
        return sum(1 for r in self.rows if r.get("meal_mg", 0) > 0)

    @property
    def n_boluses(self) -> int:
        return sum(1 for r in self.rows if r.get("bolus_pmol_kg", 0) > 0)

    @property
    def n_basal_segments(self) -> int:
        return len(self.basal_segments)


class BaseParser(ABC):
    name: str

    @classmethod
    @abstractmethod
    def can_handle(cls, source: str, fmt: str = "") -> bool:
        ...

    @abstractmethod
    def scan(self, source_path: str, study_filter: Optional[str] = None,
             limit: Optional[int] = None) -> dict[str, dict]:
        ...

    @abstractmethod
    def parse_patient(self, patient_id: str, source_path: str,
                      weight_kg: Optional[float] = None,
                      start_date: Optional[str] = None,
                      days: int = 0) -> PatientData:
        ...

    @abstractmethod
    def check_window(self, source_path: str, patient_id: str, days: int,
                     min_meals: int, min_boluses: int,
                     min_cgm_points: int) -> tuple[bool, int, int, int]:
        ...

    # ── Shared utilities ──────────────────────────────────────────────

    @staticmethod
    def _parquet_signature(path: str) -> dict | None:
        try:
            st = os.stat(path)
            return {"size": st.st_size, "mtime": st.st_mtime_ns}
        except OSError:
            return None

    @staticmethod
    def _load_scan_cache(cache_dir: str, sig: dict | None,
                         study: str | None) -> dict | None:
        cache_path = os.path.join(cache_dir, SCAN_CACHE_FILE)
        if not os.path.exists(cache_path) or sig is None:
            return None
        try:
            with open(cache_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        stored = data.get("parquet_signature")
        if stored is None or stored.get("size") != sig["size"] or stored.get("mtime") != sig["mtime"]:
            return None
        if data.get("study") != study:
            return None
        patients = data.get("patients")
        if patients is None:
            return None
        print(f"  Loaded {len(patients)} patients from cache ({cache_path})")
        return patients

    @staticmethod
    def _save_scan_cache(cache_dir: str, patients: dict,
                         sig: dict | None, study: str | None):
        if sig is None:
            return
        cache_path = os.path.join(cache_dir, SCAN_CACHE_FILE)
        data = {"parquet_signature": sig, "study": study, "patients": patients}
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cache_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, cache_path)
        except OSError:
            print(f"  WARN: failed to write scan cache to {cache_path}", file=sys.stderr)

    @staticmethod
    def report_by_study(patients: dict, output_path: str | None = None):
        sorted_patients = sorted(patients.values(), key=lambda p: p["id"])

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fieldnames = ["id", "source_file", "weight_lbs", "rows",
                          "cgm", "meals", "boluses"]
            with open(output_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(sorted_patients)

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

        if output_path:
            print(f"\nFull report saved to {output_path}")

    @staticmethod
    def check_csv_quality(cgm_csv: str, basal_csv: str) -> dict:
        result = dict(cgm_count=0, meal_count=0, bolus_count=0,
                      basal_segments=0, total_min=0, has_data=False,
                      cgm_density=0.0)
        if not os.path.exists(cgm_csv) or not os.path.exists(basal_csv):
            return result
        with open(cgm_csv, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    t = int(float(row["t_min"]))
                except (KeyError, ValueError):
                    continue
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
        result["has_data"] = result["cgm_count"] > 0 and result["basal_segments"] >= 1
        result["cgm_density"] = result["cgm_count"] / max(1, result["total_min"] / 5)
        return result

    @staticmethod
    def write_csvs(data: PatientData, output_dir: str,
                   suffix: str | None = None) -> tuple[str, str]:
        if suffix is None:
            ndays = data.total_min // DAY_MINUTES + 1
            suffix = f"{ndays}d"
        prefix = f"patient{data.patient_id}_{suffix}"

        os.makedirs(output_dir, exist_ok=True)

        cgm_path = os.path.join(output_dir, f"{prefix}_cgm_meals_bolus.csv")
        with open(cgm_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t_min", "cgm_mgdl", "meal_mg", "bolus_pmol_kg"])
            w.writeheader()
            w.writerows(data.rows)
        print(f"  {data.n_cgm} CGM, {data.n_meals} meals, {data.n_boluses} boluses -> {cgm_path}")

        basal_path = os.path.join(output_dir, f"{prefix}_basal.csv")
        with open(basal_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t_min", "basal_u_per_hr"])
            w.writeheader()
            w.writerows(data.basal_segments)
        print(f"  {data.n_basal_segments} basal segments -> {basal_path}")

        return cgm_path, basal_path
