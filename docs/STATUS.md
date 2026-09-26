# Project status (2026-09)

Structural and practical identifiability of the UVA-Padova model from a
single CGM channel: done and audited. Online adaptation (CKF/RTS/meal
priors) compiles and runs, without real-data validation. No MPC controller.
Not a medical device.

## Done

- **Reduced T1DM model** `include/uva_padova/types.hpp:56` (14 states) from
  Dalla Man 2007/2013/2018, with T1D fixes (HE_b, S≡0, single-hormone).
- **Structural identifiability** `real_world_validation/identifiability/` —
  exhaustive analysis with `StructuralIdentifiability.jl`, CSVs committed.
  Core finding: scaling group `k_p3 ↔ V_m0 ↔ V_G ↔ (F_cns - k_p1)`; fixing
  `K_m0, V_I, BW, F_cns, HE_b` restores local identifiability of 21 params.
  Empirically it narrows to 4–5 active params.
- **Offline identification** `include/uva_padova/cost_function.hpp:97`,
  `optimizer_impl.hpp:50` — CMA-ES (global) → LM (Ceres), per-element
  SoftL1, log-space `θ`, `SimulationCache`, sandwich covariance. Gates A
  (DBSCAN multiplicity), B (Ljung-Box + regime bias), C (FIM+prior SVD) in
  `src/diagnostics.cpp`.
- **Synthetic recovery study** `synthetic_study/` — virtual patients with
  heteroskedastic noise, dropouts, meal/bolus errors
  (`synthetic_data_generator.hpp:76`).
- **Tests** `tests/unit/` + `tests/integration/` — adjustments, PriorCost,
  ObservationCost, optimizer, CKF.

## Not validated

- **CKF online layer** (`cubature_kalman_filter.hpp:60`,
  `system_model.hpp:25`, `regime_modulated_q.hpp`,
  `online_observability.hpp`, `rts_smoother.hpp`, `meal_prior_store.hpp`).
  Compiles, `synthetic_study/online_evaluate.cpp` runs, but there is no
  real-data closed-loop validation and no mobile profiling. Open items:
  #9 heteroskedasticity weights, #18 mealD fragility, #22 Cholesky
  hardening, #26 meal P-jump heuristics, #28 finite-diff scaling —
  see [ROADMAP.md](../ROADMAP.md), full history in [audit-2026-07.md](audit-2026-07.md).
- **MPC controller** — spec only (`docs/main.md` §1.2), no code.

## Notes

- `real_world_validation/` needs the T1DEXI/OhioT1DM parquet downloaded
  separately (`UVA_PADOVA_PARQUET`); no patient data is committed.
- Large sweep outputs (megabytes of JSON, hours to recompute) live outside
  the repo; `main` builds on clean `ubuntu-latest` + `macos-14` and the
  fast suite passes.
