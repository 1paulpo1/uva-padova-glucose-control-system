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
# outputs: examples/complex_irregular_patient/output/{synth.json,figs/}
```

Runtime is ~1–3 h. Results are **git-ignored by design** — only the script
and this doc are committed.

## What "good" looks like

- `figs/true_vs_recovered.png`: per-panel titles report `|err|` in log-space;
  axes never zoom tighter than ±0.25, so closeness to the red `y=x` is read
  against a fixed scale (see `plot_true_vs_recovered(min_half_span=0.25)`).
  Mean `|err|` is in the figure suptitle.
- `figs/residuals_p0.png`: observed vs predicted CGM overlap, residuals
  within ±10 mg/dL, flat ACF (Gate B passes).
- `synth.json`: `gate_a.passed=true` (single CMA-ES cluster),
  `gate_b.passed=true`.

## Why this case is informative

Correction boluses decorrelate insulin action from meals, and day-level meal
randomness excites GI kinetics — the two ingredients that make `kabs/kgri`
and `p2U` observable through a single CGM channel (cf. `docs/main.md` §2.11,
§3.4).
