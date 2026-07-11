#!/usr/bin/env python3
"""
Extract patient metadata from the OhioT1DM/T1DEXI parquet file.
Output: validation/patients_meta.csv
"""
import pyarrow.parquet as pq
import pandas as pd

PARQUET_PATH = r"C:\Users\beleg\Downloads\metabonet_public.parquet"
OUTPUT_PATH = r"validation\patients_meta.csv"

static_cols = [
    "id", "weight", "height", "age", "gender", "cgm_device",
    "insulin_type_basal", "insulin_type_bolus", "insulin_delivery_algorithm",
    "insulin_delivery_device", "insulin_delivery_modality", "is_pregnant",
    "is_test", "source_file", "treatment_group", "age_of_diagnosis",
    "ethnicity", "subject_split_across_traintest",
]

pf = pq.ParquetFile(PARQUET_PATH)
all_meta = {}
n_groups = pf.metadata.num_row_groups
print(f"Scanning {n_groups} row groups for metadata...")

for rg_idx in range(n_groups):
    t = pf.read_row_group(rg_idx, columns=static_cols)
    df = t.to_pandas()
    for pid in df["id"].unique():
        if pid in all_meta:
            continue
        sub = df[df["id"] == pid]
        row = {"id": pid}
        for col in static_cols[1:]:
            non_null = sub[col].dropna()
            if len(non_null) > 0:
                row[col] = non_null.iloc[0]
        all_meta[pid] = row
    if (rg_idx + 1) % 20 == 0:
        print(f"  RG {rg_idx+1}/{n_groups}: {len(all_meta)} unique patients")

meta_df = pd.DataFrame.from_dict(all_meta, orient="index")
meta_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")
print(f"Total patients: {len(meta_df)}")
print(f"Weight available: {meta_df['weight'].notna().sum()}/{len(meta_df)}")
print(f"Sample rows:")
print(meta_df.head(10).to_string())
