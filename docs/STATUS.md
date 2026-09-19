# Project Status & Honest Scope (2026-09)

**TL;DR:** The structural + practical identifiability of the UVA-Padova model by a single CGM channel is solid and audited. Online adaptation (CKF/RTS/meal-priors) compiles and runs but is **experimental, not validated**. MPC is **not implemented** — listed as Future Work. This is a research prototype, not a medical device.

## What is done and portfolio-ready

- **Reduced T1DM model** `include/uva_padova/types.hpp:56` (14 states) derived from Dalla Man 2007/2013/2018, with T1D fixes (HE_b, S≡0, single-hormone).
- **Structural identifiability** `real_world_validation/identifiability/` — Julia `StructuralIdentifiability.jl` exhaustive analysis, CSVs committed. Core finding: scaling group `k_p3 ↔ V_m0 ↔ V_G ↔ (F_cns - k_p1)`; fixing `K_m0, V_I, BW, F_cns, HE_b` restores local identifiability of 21 params. Empirically narrows to 4-5 active params.
- **Offline identification** `include/uva_padova/cost_function.hpp:97 + optimizer_impl.hpp:50` — CMA-ES (global) → LM (Ceres) with per-element SoftL1 (fix #7), log-space `θ`, `SimulationCache`, sandwich covariance. Gates A (DBSCAN multiplicity), B (Ljung-Box + regime bias), C (FIM+prior SVD) in `src/diagnostics.cpp`.
- **Synthetic recovery study** `synthetic_study/` — virtual patients with heteroskedastic noise, dropouts, meal/bolus errors (`synthetic_data_generator.hpp:76`). Recovery error on 300-patient sweeps (see archived results in `~/Documents/uva-padova-archive-local-backup/`). Plots in `synthetic_study/analyze_results.py` (true vs recovered, error boxplots, Gate dashboards).
- **Tests** `tests/unit/` + `tests/integration/` — adjustments, PriorCost, ObservationCost, optimizer, CKF.

## What is experimental / not validated

- **CKF online layer** `include/uva_padova/cubature_kalman_filter.hpp:60`, `system_model.hpp:25` (SS caching fix #16), `regime_modulated_q.hpp`, `online_observability.hpp` (Gramian fix #27), `rts_smoother.hpp` (Jacobian fix #23), `meal_prior_store.hpp` (context fix #33). Compiles, `synthetic_study/online_evaluate.cpp` runs, but **no real-data closed-loop validation** and no mobile profiling. `FIXME.md` still lists #9 heteroskedasticity weights, #18 mealD fragility, #24-25 Q/P0 scaling (partially fixed), #28 finite-diff scaling.
- **MPC controller** — spec only (`docs/main.md:1.2`, `docs/STATUS.md`). No code. Listed under Future Work.

## What is future work

1. MPC with safety constraints (hypoglycemia), uncertainty-aware cost.
2. Heteroskedastic weights `w_k` in `ObservationCost`, real-data online validation, on-device benchmarks (1.2 mobile constraint).
3. Dataset license check — `real_world_validation/` needs T1DEXI/OhioT1DM parquet downloaded via `UVA_PADOVA_PARQUET`; no patient data is committed (see `.gitignore`).
4. Clean git history — large `synthetic_study/**/100-*.json` (5-18M) removed from index, kept locally (`~/Documents/uva-padova-archive-local-backup/`). History rewrite via `git filter-repo` planned before first public tag.

## How to read results without re-running 10h sweeps

Archived sweeps (300 patients, 3 days, 5 restarts) are **not in git** but kept locally:
```
/Users/paul/Documents/uva-padova-archive-local-backup/synthetic_study/
```
Keep them to build the “Achievements” section in README without recomputing. For CI, a tiny `--patients 3 --evals 200` smoke run is enough.

## Branching

- `main` — portfolio-ready, builds on clean `ubuntu-latest` + `macos-14`, tests pass.
- Large artifacts are `.gitignore`’d; `git add -f` a single small example JSON if you need a demo in README.
