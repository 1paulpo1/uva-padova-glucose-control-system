# Complex irregular patient (showcase example)

Single virtual patient, 1 day, with **irregular meals** and **correction
boluses** (the excitation that makes single-channel identification possible):

- **Irregular meals**: `--n-meals-mean 4 --n-meals-std-day 1` (count varies),
  `--split-cv 0.3 --time-cv 0.12` (size/timing jitter)
- **Correction boluses**: `--correction-bolus-prob 1` (≈1/hour Poisson),
  mean 1 U ± 0.5 U — insulin without meals, breaks meal/bolus collinearity
- **Thorough optimizer**: 25 CMA-ES restarts × 500 evals + LM refinement

Two acts: the identifiable core recovers fully; the extended set documents
the failure mode. Same patient setup, different param configs.

## Run

```bash
./examples/complex_irregular_patient/run.sh          # identifiable core (4 params)
# outputs: examples/complex_irregular_patient/output/{synth.json,synth.csv,figs/}
./examples/complex_irregular_patient/run_stats.sh 100 123   # bulk stats on the core
# outputs: examples/complex_irregular_patient/output/stats_100/{stats.json,stats.csv,figs/}
```

Runtime ~1 min for the core script on a laptop
(measured 47 s on MacBook Pro for 25×500). Results are
**git-ignored by design** — only scripts and this doc are committed.

## Act 1 — the identifiable core (reference outcome)

Config `config/default_config_5_active.json`: 4 params
(`VG, VI, kp1, Gb`), seed 42, 25×500. Measured:

- `success=YES`, LM converges 104.2 → 104.1683 in 6 iters (~12.5k evals).
- **Gate A PASS** (1 cluster — all restarts agree),
  **Gate B PASS** (residuals white, no regime bias),
  **Gate C: 0 prior-dominated**, mean `|θ_err|` ≈ 0.018 log-units
  (`VG` 1.5%, `VI` 3.9%, `kp1` 1.3%, `Gb` 0.5%).
- `figs/true_vs_recovered.png`: all four points on the red `y=x`
  (per-panel `|err|`, fixed ±0.25 scale); `figs/residuals_p0.png`: tight
  fit, flat ACF.

This is the publishable positive: the full core `{VG, VI, kp1, Gb}` is
recovered from one day of noisy single-channel CGM. `run_stats.sh`
scales this to N patients for sufficient statistics.

**At scale (300 patients, same setup, `output/300-5p-new.json` — local file,
not in git):** 300/300 success; gate A 87.3% (262/300), gate B 93.0%
(279/300); median `|err|` per param `VG` 0.005, `VI` 0.008, `kp1` 0.006,
`Gb` 0.003 (all <1% in physical units). Gate C averages 0.2 dominated
params per patient (239/300 clean); `VI` is flagged most often (46×) —
unsurprising, insulin kinetics from glucose alone is the weakest link of
the four. Three nuances worth stating plainly:
(a) the 38 gate-A failures are almost entirely 2-cluster splits, and only
1 patient fails A and B together — the structure (B) holds wherever
consensus (A) wobbles;
(b) on this easy set, gate passage barely separates error: median
worst-param error is 0.014 for pass vs 0.015 for fail — the gates are
calibrated for the hard regime and read conservative here;
(c) the few large misses are isolated per param, not per patient
(worst `VI` pid 170, worst `Gb` pid 147, worst `kp1` pid 117) — no single
broken patient, just the tail of the noise. Aggregate plots:
`output/figs-300/` (same registry, same fixed scales).

## Act 2 — the extended set (documented failure mode)

Config `config/complex_irregular_15_active.json`: 15 params
(distribution `VG/VI`, GI kinetics `k1/k2/kmax/kmin/kabs/kgri/b`,
EGP `kp1`, utilization `p2U`, SC insulin `ka1/ka2`, sensor `Ts`,
basal `Gb`). Measured reference (preceding 16-param revision, seed 42):

- `success=YES`, LM converges 432.9 → 103.6 (35 iters, ~11k evals).
- **Gate A FAIL** (25 restarts scatter, 0 clusters — no consensus),
  **Gate B PASS** (residuals white, no regime bias),
  **Gate C: 8/16 prior-dominated**
  (`kmax/kmin/kabs/kgri/d/ki/p2U/Ts`), mean `|θ_err|` ≈ 0.21 log-units.
- Split by identifiability: `Gb/kp1/k1/kmin/ka2/b` recover within 5%,
  `k2/kmax/ka1` within ~12%, `kabs/kgri/p2U` miss by ~20%,
  `d/kp2/ki/Ts` miss by 40–97%. The table in the console output and
  `figs/true_vs_recovered.png` show exactly which params the day resolves
  and which it does not — which is how `d/kp2/ki` got fixed in the current
  15-param revision.
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
