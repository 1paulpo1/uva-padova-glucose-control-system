from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pyarrow
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .base_parser import BaseParser, PatientData, U_TO_PMOL, DAY_MINUTES
from .registry import register

AUDIT_COLS = ["id", "weight", "source_file", "carbs", "bolus", "CGM"]


def _aggregate_basal(df):
    if df.empty:
        return []
    segments = []
    seg_start = df["date"].iloc[0]
    prev_rate = df["basal"].iloc[0]
    if prev_rate is None or (isinstance(prev_rate, float) and prev_rate != prev_rate):
        prev_rate = 0.0
    for idx, row in df.iterrows():
        rate = row["basal"]
        if rate is None or (isinstance(rate, float) and rate != rate):
            rate = 0.0
        if abs(rate - prev_rate) > 1e-9:
            segments.append((seg_start, prev_rate))
            seg_start = row["date"]
            prev_rate = rate
    segments.append((seg_start, prev_rate))
    return segments


def _minutes_since_ref(dt: datetime, ref: datetime) -> int:
    return int((dt - ref).total_seconds() // 60)


@register
class MetabonetParser(BaseParser):
    """Parser for the unified T1DEXI/OhioT1DM parquet dataset."""

    name = "metabonet"

    @classmethod
    def can_handle(cls, source: str, fmt: str = "") -> bool:
        _, ext = os.path.splitext(source)
        if ext.lower() != ".parquet":
            return False
        if fmt in ("parquet", "metabonet", "t1dexi"):
            return True
        try:
            pf = pq.ParquetFile(source)
            required = {"id", "date", "CGM", "basal"}
            schema_names = set(pf.schema_arrow.names)
            return required.issubset(schema_names)
        except Exception:
            return False

    def _resolve_weight(self, source_path: str, patient_id: str,
                        fallback_kg: float | None) -> tuple[float, float] | None:
        if fallback_kg is not None:
            return (fallback_kg, fallback_kg)
        try:
            t = pq.read_table(
                source_path,
                columns=["id", "weight"],
                filters=[("id", "=", patient_id)],
            )
            col = t.column("weight")
            valid = pc.drop_null(col)
            if len(valid) > 0:
                w_lbs = valid[0].as_py()
                if w_lbs is not None and w_lbs > 0:
                    return (w_lbs, w_lbs * 0.453592)
        except (OSError, pyarrow.ArrowException):
            pass
        return None

    def scan(self, source_path: str, study_filter: Optional[str] = None,
             limit: Optional[int] = None) -> dict[str, dict]:
        pf = pq.ParquetFile(source_path)
        n_groups = pf.metadata.num_row_groups
        n_scan = min(n_groups, max(40, 100)) if limit is None else n_groups
        # Cap scan to 60 row groups to bound memory usage
        if n_scan > 60:
            n_scan = min(n_scan, 60)

        patients: dict = {}
        done = False
        for rg_idx in range(n_scan):
            if done:
                break
            t = pf.read_row_group(rg_idx, columns=AUDIT_COLS)
            df = t.to_pandas()
            if study_filter is not None:
                df = df[df["source_file"] == study_filter]
            if df.empty:
                continue
            for pid in df["id"].unique():
                if pid in patients:
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
                    continue
                patients[pid] = {
                    "id": pid,
                    "weight_lbs": w.iloc[0],
                    "source_file": sub["source_file"].iloc[0],
                    "rows": len(sub),
                    "meals": int((sub["carbs"].fillna(0) > 0).sum()),
                    "boluses": int((sub["bolus"].fillna(0) > 0).sum()),
                    "cgm": int(sub["CGM"].notna().sum()),
                }
                if limit is not None and len(patients) >= limit:
                    done = True
                    break
            if (rg_idx + 1) % 20 == 0:
                print(f"  RG {rg_idx+1}/{n_scan}: {len(patients)} patients", flush=True)
        return patients

    def check_window(self, source_path: str, patient_id: str, n_days: int,
                     min_meals: int, min_boluses: int,
                     min_cgm_points: int = 50) -> tuple[bool, int, int, int]:
        try:
            t = pq.read_table(
                source_path,
                columns=["id", "date", "carbs", "bolus", "CGM"],
                filters=[("id", "=", patient_id)],
            )
        except Exception:
            return (False, 0, 0, 0)
        if len(t) == 0:
            return (False, 0, 0, 0)
        df = t.to_pandas()
        df["date"] = df["date"].dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
        ref = df["date"].iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
        t_end = n_days * 1440
        df["t_min"] = (df["date"] - ref).dt.total_seconds() / 60
        window = df[(df["t_min"] >= 0) & (df["t_min"] < t_end)]
        meals = int((window["carbs"].fillna(0) > 0).sum())
        boluses = int((window["bolus"].fillna(0) > 0).sum())
        cgm_count = int(window["CGM"].notna().sum())
        return (meals >= min_meals and boluses >= min_boluses and cgm_count >= min_cgm_points,
                meals, boluses, cgm_count)

    def parse_patient(self, patient_id: str, source_path: str,
                      weight_kg: Optional[float] = None,
                      start_date: Optional[str] = None,
                      days: int = 0) -> PatientData:
        pid = str(patient_id)

        weight_res = self._resolve_weight(source_path, pid, weight_kg)
        if weight_res is None:
            raise ValueError(f"No weight found for patient {pid} and no --weight given")
        w_lbs, w_kg = weight_res

        table = pq.read_table(
            source_path,
            columns=["id", "date", "CGM", "basal", "bolus", "carbs"],
            filters=[("id", "=", pid)],
        )
        if len(table) == 0:
            raise ValueError(f"No data for patient {pid}")

        df = table.to_pandas()
        df["date"] = df["date"].dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)

        first_dt = df["date"].iloc[0]
        if start_date:
            ref_time = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            ref_time = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        if days > 0:
            t_end = days * DAY_MINUTES
        else:
            max_dt = df["date"].iloc[-1]
            t_end = _minutes_since_ref(max_dt, ref_time) + 1
            t_end = ((t_end + DAY_MINUTES - 1) // DAY_MINUTES) * DAY_MINUTES

        df["t_min"] = df["date"].apply(lambda d: _minutes_since_ref(d, ref_time))
        df = df[(df["t_min"] >= 0) & (df["t_min"] < t_end)].copy()
        if df.empty:
            raise ValueError(f"No data for patient {pid} in the specified window")

        rows = []
        for _, row in df.iterrows():
            cgm = row["CGM"]
            cgm_val = cgm if (cgm is not None and cgm == cgm and cgm > 0) else 0.0
            carbs_g = row["carbs"]
            meal_mg = 0.0
            if carbs_g is not None and carbs_g == carbs_g and carbs_g > 0:
                meal_mg = carbs_g * 1000.0
            bolus_u = row["bolus"]
            bolus_pmol_kg = 0.0
            if bolus_u is not None and bolus_u == bolus_u and bolus_u > 0:
                bolus_pmol_kg = round(bolus_u * U_TO_PMOL / w_kg, 4)
            rows.append({
                "t_min": row["t_min"],
                "cgm_mgdl": cgm_val,
                "meal_mg": meal_mg,
                "bolus_pmol_kg": bolus_pmol_kg,
            })

        df_basal = df.copy()
        df_basal["basal"] = (df_basal["basal"] * BASAL_SCALE).round(6)
        basal_segments_raw = _aggregate_basal(df_basal)
        basal_segments = []
        for dt, rate in basal_segments_raw:
            t = _minutes_since_ref(dt, ref_time)
            if 0 <= t < t_end:
                basal_segments.append({"t_min": t, "basal_u_per_hr": rate})

        rows.sort(key=lambda r: r["t_min"])
        basal_segments.sort(key=lambda r: r["t_min"])

        ndays = t_end // DAY_MINUTES
        total_min = ndays * DAY_MINUTES

        return PatientData(
            patient_id=pid,
            weight_kg=w_kg,
            source="metabonet",
            total_min=total_min,
            rows=rows,
            basal_segments=basal_segments,
        )
