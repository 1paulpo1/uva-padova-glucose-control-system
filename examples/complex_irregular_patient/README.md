# Complex irregular patient (showcase example)

Single virtual patient, 1 day, designed to be **hard but solvable**:

- **Irregular meals**: `--n-meals-mean 4 --n-meals-std-day 1` (count varies),
  `--split-cv 0.3 --time-cv 0.12` (size/timing jitter)
- **Correction boluses**: `--correction-bolus-prob 1` (≈1/hour Poisson),
  mean 1 U ± 0.5 U — insulin without meals, breaks meal/bolus collinearity
- **16 active params** (`config/complex_irregular_16_active.json`):
  GI kinetics (`k1/k2/kmax/kmin/kabs/kgri/b/d`), EGP (`kp1/kp2/ki`),
  utilization (`p2U`), SC insulin (`ka1/ka2`), sensor (`Ts`), basal (`Gb`)
- **Thorough optimizer**: 25 CMA-ES restarts × 500 evals + LM refinement

## Run

```bash
./examples/complex_irregular_patient/run.sh
# outputs: examples/complex_irregular_patient/output/{synth.json,synth.csv,figs/}
```

Runtime is a few minutes on a laptop (the script's 25×500 takes ~1–2 min;
a heavier 50×1000 variant measured 240 s on MacBook Pro — cost scales with
restarts × evals plus LM refinement). Results are **git-ignored by design** —
only the script and this doc are committed.

## What to expect (honest)

This is a deliberately hard case — 16 params from a single day of CGM —
so read it as a pipeline exercise, not a victory lap. A typical outcome:

- `success=YES`, LM improves on the best CMA-ES restart
  (e.g. 213 → 123), but **Gates A/B fail** (scattered restarts, residual
  autocorrelation) and Gate C marks roughly half the params
  prior-dominated, with mean `|θ_err|` ≈ 0.3–0.5 log-units.
- `figs/true_vs_recovered.png`: per-panel titles report `|err|` in log-space;
  axes never zoom tighter than ±0.25, so closeness to the red `y=x` is read
  against a fixed scale (see `plot_true_vs_recovered(min_half_span=0.25)`).
  Mean `|err|` is in the figure suptitle. Before this fix, matplotlib
  auto-scaled each panel to a microscopic range and any point looked "on the
  line" — the old visual impression of a good fit was a scale artifact.
- `figs/residuals_p0.png`: observed vs predicted CGM plus residuals and ACF;
  the primary readout alongside the JSON numbers.

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
