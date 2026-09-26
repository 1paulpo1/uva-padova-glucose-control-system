#!/usr/bin/env bash
# Single-patient run on the identifiable core (VG, VI, kp1, Gb):
# irregular meals + correction boluses, 1 day, seed 42.
# Reference outcome: Gates A/B PASS, 0 prior-dominated, mean |err| ~0.02.
#
# Runtime ~1 min on a laptop (25 CMA-ES restarts x 500 evals, 4 params;
# measured 47 s on MacBook Pro).
# Outputs go to ./output/ (not tracked by git).
# For bulk statistics on the core see run_stats.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$(dirname "${BASH_SOURCE[0]}")/output"
mkdir -p "$OUT_DIR"

"$REPO_ROOT/build/synthetic_study/cross_validate" \
  --patients 1 \
  --seed 42 \
  --restarts 25 \
  --evals 500 \
  --params-config "$REPO_ROOT/config/default_config_5_active.json" \
  --duration-min 1440 \
  --n-meals-mean 4 \
  --n-meals-std-day 1 \
  --correction-bolus-prob 1 \
  --correction-bolus-mean 1 \
  --correction-bolus-std 0.5 \
  --split-cv 0.3 \
  --time-cv 0.12 \
  --noise-preset cgm \
  --csv "$OUT_DIR/synth.csv" \
  --json "$OUT_DIR/synth.json" \
  --verbose

python3 "$REPO_ROOT/synthetic_study/analyze_results.py" \
  --json "$OUT_DIR/synth.json" \
  --out "$OUT_DIR/figs"

echo "Done. See $OUT_DIR/figs/true_vs_recovered.png and $OUT_DIR/figs/residuals_p0.png"
