# Complex irregular patient (showcase example)

Single virtual patient, 1 day, designed to be **hard but solvable**:

- **Irregular meals**: `--n-meals-mean 4 --n-meals-std-day 1` (count varies),
  `--split-cv 0.3 --time-cv 0.12` (size/timing jitter)
- **Correction boluses**: `--correction-bolus-prob 1` (≈1/hour Poisson),
  mean 1 U ± 0.5 U — insulin without meals, breaks meal/bolus collinearity
- **15 active params** (`config/complex_irregular_15_active.json`):
  distribution (`VG/VI`), GI kinetics (`k1/k2/kmax/kmin/kabs/kgri/b`,
  wider prior 2.0 on `kmax/kmin/kabs/kgri`), EGP (`kp1`),
  utilization (`p2U`), SC insulin (`ka1/ka2`), sensor (`Ts`, prior 2.0),
  basal (`Gb`). `d/kp2/ki` are fixed — the preceding 16-param revision
  showed them unrecoverable from a single day (83/97/57% off,
  all Gate-C-dominated).
- **Thorough optimizer**: 25 CMA-ES restarts × 500 evals + LM refinement

## Run

```bash
./examples/complex_irregular_patient/run.sh
# outputs: examples/complex_irregular_patient/output/{synth.json,synth.csv,figs/}
```

Runtime is ~2 min on a laptop for the script's 25×500
(measured 119 s on MacBook Pro; a heavier 50×1000 variant took 240 s —
cost scales with restarts × evals plus LM refinement). Results are
**git-ignored by design** — only the script and this doc are committed.

## What to expect (honest)

This is a deliberately hard case — 15 params from a single day of CGM —
so read it as a pipeline exercise, not a victory lap. Measured reference
run below used the preceding 16-param config revision (seed 42, 25×500);
re-run `./run.sh` (~2 min) to refresh the numbers for the current 15-param
revision, which is expected to do better (three worst offenders fixed,
wider priors where the data needs room):

- `success=YES`, LM converges 432.9 → 103.6 (35 iters, ~11k evals).
- **Gate A FAIL** (25 restarts scatter, 0 clusters — no consensus),
  **Gate B PASS** (residuals white, no regime bias),
  **Gate C: 8/16 prior-dominated**
  (`kmax/kmin/kabs/kgri/d/ki/p2U/Ts`), mean `|θ_err|` ≈ 0.21 log-units.
- Split by identifiability: `Gb/kp1/k1/kmin/ka2/b` recover within 5%,
  `k2/kmax/ka1` within ~12%, `kabs/kgri/p2U` miss by ~20%,
  `d/kp2/ki/Ts` miss by 40–97%. The table in the console output and
  `figs/true_vs_recovered.png` (per-panel `|err|`, fixed ±0.25 scale) show
  exactly which params the day resolves and which it does not.
- `figs/residuals_p0.png`: observed vs predicted CGM plus residuals and ACF
  (lag-domain bottom panel).

## FAQ

**Why is `cost(θ=true)=105.8` above the optimum `103.6`?**
Expected, not a bug. The cost is measured against *noisy* observations:
even the ground truth leaves exactly the sensor noise as residuals
(`cost(true)` ≈ noise floor). The 16-param estimate overfits that noise
slightly (Δ≈2 on a scale of ~100 with 289 residuals — statistical zero).
A *large* negative gap would be the alarming sign (gross overfit or a bug);
the pipeline's sanity check encodes this
(`cross_validate.cpp`: clean-data runs must satisfy `cost(true)≈0`).

**Why does the CGM fit look perfect while half the params are wrong?**
`residuals_p0.png` compares prediction against the *noisy observations the
model was fit to* — overlap is guaranteed and proves nothing about
parameters. Parameter quality lives in the per-param table, in
`true_vs_recovered.png` (readable since the fixed-scale fix — before it,
autoscale made every point look "on the line"), and in the Gates:
B PASS says the structure is right, A FAIL + C say one day doesn't pin
16 params. That combination is the whole point of the example.

## Environment note

PCA/t-SNE and statsmodels-ACF plots are optional: if your env has a broken
combo (e.g. sklearn built for NumPy 1.x running under NumPy 2.x), they are
skipped with a warning and the analysis continues — core plots
(`true_vs_recovered`, `residuals`, `error_boxplot`, `gate_summary`) need only
numpy + matplotlib. The manual-ACF fallback covers a missing statsmodels.

## Why this case is informative

Correction boluses decorrelate insulin action from meals, and day-level meal
randomness excites GI kinetics — the two ingredients that make `kabs/kgri`
and `p2U` observable through a single CGM channel (cf. `docs/main.md` §2.11,
§3.4).
