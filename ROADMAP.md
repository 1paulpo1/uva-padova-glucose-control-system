# Roadmap

Current scope and honesty statement: [docs/STATUS.md](docs/STATUS.md).
Full historical audit (RU, frozen July 2026): [docs/audit-2026-07.md](docs/audit-2026-07.md).
Terminology (`validation` vs `crossval` rename): [docs/TERMINOLOGY.md](docs/TERMINOLOGY.md).

## Done (identifiability core)

- Reduced 14-state T1DM model from Dalla Man 2007/2013/2018.
- Structural identifiability study (`real_world_validation/identifiability/`).
- Offline identification CMA-ES → LM with per-element SoftL1, sandwich
  covariance, Gates A/B/C (`audit` items 1, 2, 4–7, 10, 11, 13, 15–17, 23–25,
  27, 30, 33, 36 resolved; 8, 12, 14, 21, 31, 32, 34, 35 closed as
  false-positive/spec-clarification).
- Curated showcase: `examples/complex_irregular_patient/` (15 params, 1 day).

## Known limitations (from the audit, still open)

- **#9 — homoscedastic CGM noise.** `ObservationCost` divides all residuals
  by a single σ=2.9; the spec asks for heteroskedastic weights
  `w_k ∝ 1/σ̂_k²` (error grows at glycemic extremes).
- **#18 — `mealD` lifetime.** Previous-meal mass persists in state and can
  mix with leftovers in `gastric_emptying_rate`. Works in practice (mealD
  and Qsto1 are set together) but fragile.
- **#19/#20 — ODE helper hygiene.** Per-block helpers re-run the full `rhs()`
  (~7× redundant); `egp()` duplicates the inline formula. Perf/cleanliness,
  not correctness.
- **#22 — CKF Cholesky without SPD guard.** No `llt().info()` checks in
  predict/update/RTS. Latent numerical risk, hardening task.
- **#26 — meal P-jump heuristics.** `P += CHO·{0.05, 0.01}` has no
  uncertainty grounding.
- **#28/#29 — observability cost.** Unscaled finite-diff epsilon; ~23 full
  ODE solves per call. Fine offline, too heavy for on-device real time.
- **#3 (partial) — online set ≠ spec set.** `online_prod.json` covers
  `{VG, kabs, kp1, kp2, p2U, Gb}`; spec additionally lists `r1` (only
  observable in hypoglycemia) and aggregated `η_ISF`.
- **Flaky test (new, not in audit).**
  `RecoveryBenchmark.oneday_moderate_noise` intermittently dies with bus
  error in the LM phase (suspected stack pressure from `Jet<64>` autodiff
  in Ceres threads — not investigated). Quarantined from CI via
  `-E RecoveryBenchmark.oneday_moderate_noise`; fast suite is 97/97 green.

## Future work

1. MPC with hypoglycemia safety constraints + uncertainty-aware cost
   (spec-only today, `docs/main.md` §1.2).
2. Real-data closed-loop online validation + on-device profiling (§1.2
   mobile constraint).
3. Heteroskedastic weights (#9) + observability perf (#28/#29) as
   prerequisites for (2).
4. Root-cause the RecoveryBenchmark bus error; re-enable in CI.
