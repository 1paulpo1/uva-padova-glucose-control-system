from __future__ import annotations

import copy
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from parsers.base_parser import PatientData


@dataclass
class FilterConfig:
    gb_method: str = "median"
    min_cgm_points: int = 50
    min_basal_segments: int = 2
    max_cgm_gap_minutes: float = 120.0
    outlier_zscore: float = 4.0


@dataclass
class PatientMetadata:
    patient_id: str
    study: str
    gb_estimate: float
    n_cgm: int
    n_meals: int
    n_boluses: int
    n_basal_segments: int
    total_minutes: int
    ndays: float
    meals_per_day: float
    boluses_per_day: float
    cgm_density: float
    max_gap_minutes: float
    n_outliers: int
    warnings: list[str] = field(default_factory=list)
    valid: bool = True


def _estimate_gb(rows: list[dict], method: str) -> float:
    cgm_vals = [r["cgm_mgdl"] for r in rows
                if r.get("cgm_mgdl", 0) > 0 and r["cgm_mgdl"] == r["cgm_mgdl"]]
    if not cgm_vals:
        return float("nan")
    if method == "median":
        return statistics.median(cgm_vals)
    elif method == "fasting_p10":
        cgm_vals.sort()
        idx = max(0, int(len(cgm_vals) * 0.1) - 1)
        return cgm_vals[idx]
    return statistics.median(cgm_vals)


def _max_cgm_gap(rows: list[dict]) -> float:
    """Return longest consecutive run of missing CGM (0 or NaN) in minutes."""
    max_gap = 0.0
    current_gap = 0.0
    prev_t = None
    for r in rows:
        t = r["t_min"]
        cgm = r.get("cgm_mgdl", 0)
        missing = not (cgm > 0 and cgm == cgm)
        if missing:
            if current_gap == 0.0:
                current_gap = 5.0
            else:
                current_gap += 5.0
        else:
            if current_gap > max_gap:
                max_gap = current_gap
            current_gap = 0.0
        prev_t = t
    if current_gap > max_gap:
        max_gap = current_gap
    return max_gap


def _count_outliers(rows: list[dict], window: int, zscore: float) -> int:
    """Count CGM points that deviate >zscore MAD from rolling median."""
    cgm_vals = []
    times = []
    for r in rows:
        v = r.get("cgm_mgdl", 0)
        if v > 0 and v == v:
            cgm_vals.append(v)
            times.append(r["t_min"])

    if len(cgm_vals) < window // 5:
        return 0

    n_outliers = 0
    half = window // 2
    for i, t in enumerate(times):
        lo = i
        while lo > 0 and t - times[lo - 1] <= half:
            lo -= 1
        hi = i
        while hi < len(times) - 1 and times[hi + 1] - t <= half:
            hi += 1
        segment = cgm_vals[lo:hi + 1]
        if len(segment) < 5:
            continue
        med = statistics.median(segment)
        mad = statistics.median(abs(v - med) for v in segment)
        if mad < 1e-6:
            continue
        if abs(cgm_vals[i] - med) / mad > zscore:
            n_outliers += 1
    return n_outliers


def process_patient(
    data: PatientData,
    config: Optional[FilterConfig] = None,
) -> tuple[PatientData, PatientMetadata]:
    if config is None:
        config = FilterConfig()

    rows = copy.deepcopy(data.rows)
    basal_segments = copy.deepcopy(data.basal_segments)

    total_min = data.total_min
    ndays = max(1.0, total_min / 1440.0)

    n_cgm = sum(1 for r in rows if r.get("cgm_mgdl", 0) > 0 and r["cgm_mgdl"] == r["cgm_mgdl"])
    n_meals = data.n_meals
    n_boluses = data.n_boluses
    n_basal = len(basal_segments)

    warnings: list[str] = []

    # Gb estimate
    gb = _estimate_gb(rows, config.gb_method)

    # CGM density
    cgm_density = n_cgm / max(1, total_min / 5.0)

    # Max gap
    max_gap = _max_cgm_gap(rows)
    if max_gap > config.max_cgm_gap_minutes and config.max_cgm_gap_minutes > 0:
        warnings.append(
            f"CGM gap {max_gap:.0f}min > {config.max_cgm_gap_minutes:.0f}min")

    # Outliers
    n_outliers = _count_outliers(rows, 60, config.outlier_zscore)
    if n_outliers > 0:
        warnings.append(f"{n_outliers} CGM outliers (z>{config.outlier_zscore})")

    # Hard checks
    valid = True
    if n_cgm < config.min_cgm_points:
        warnings.append(f"only {n_cgm} CGM points < {config.min_cgm_points}")
        valid = False
    if n_basal < config.min_basal_segments:
        warnings.append(f"only {n_basal} basal segments < {config.min_basal_segments}")
        valid = False

    n_missing = sum(1 for r in rows
                    if not (r.get("cgm_mgdl", 0) > 0 and r["cgm_mgdl"] == r["cgm_mgdl"]))
    meals_per_day = n_meals / ndays
    boluses_per_day = n_boluses / ndays

    meta = PatientMetadata(
        patient_id=data.patient_id,
        study=data.source,
        gb_estimate=round(gb, 1),
        n_cgm=n_cgm,
        n_meals=n_meals,
        n_boluses=n_boluses,
        n_basal_segments=n_basal,
        total_minutes=total_min,
        ndays=round(ndays, 1),
        meals_per_day=round(meals_per_day, 2),
        boluses_per_day=round(boluses_per_day, 2),
        cgm_density=round(cgm_density, 4),
        max_gap_minutes=round(max_gap, 1),
        n_outliers=n_outliers,
        warnings=warnings,
        valid=valid,
    )

    filtered = PatientData(
        patient_id=data.patient_id,
        weight_kg=data.weight_kg,
        source=data.source,
        total_min=total_min,
        rows=rows,
        basal_segments=basal_segments,
    )

    return filtered, meta
