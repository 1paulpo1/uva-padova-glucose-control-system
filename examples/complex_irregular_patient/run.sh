#!/usr/bin/env bash
# Complex single-patient stress test: irregular meals + correction boluses.
# Reproduces a run where offline identification recovers all 16 params well.
#
# Runtime: a few minutes on a laptop (25 CMA-ES restarts x 500 evals,
# 16 active params; measured ~4 min for a heavier 50x1000 variant on
# MacBook Pro — scales linearly with restarts x evals plus LM refinement).
# Outputs go to ./output/ (git-ignored, never committed).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$(dirname "${BASH_SOURCE[0]}")/output"
mkdir -p "$OUT_DIR"

"$REPO_ROOT/build/synthetic_study/cross_validate" \
  --patients 1 \
  --seed 42 \
  --restarts 25 \
  --evals 500 \
  --params-config "$REPO_ROOT/config/complex_irregular_15_active.json" \
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
