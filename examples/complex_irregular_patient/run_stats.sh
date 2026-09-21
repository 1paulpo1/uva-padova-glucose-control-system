#!/usr/bin/env bash
# Bulk statistics on the identifiable core (VG, VI, kp1, Gb).
# Same virtual-patient setup as run.sh, but N patients / varying seeds.
#
# Usage: ./run_stats.sh [N_PATIENTS] [SEED]   (defaults: 100 123)
# Runtime: ~1 min per patient on a laptop (see run.sh); scale N to your box.
# Outputs go to ./output/stats_<N>/ (git-ignored, never committed).
set -euo pipefail

N="${1:-100}"
SEED="${2:-123}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$(dirname "${BASH_SOURCE[0]}")/output/stats_${N}"
mkdir -p "$OUT_DIR"

"$REPO_ROOT/build/synthetic_study/cross_validate" \
  --patients "$N" \
  --seed "$SEED" \
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
  --csv "$OUT_DIR/stats.csv" \
  --json "$OUT_DIR/stats.json"

python3 "$REPO_ROOT/synthetic_study/analyze_results.py" \
  --json "$OUT_DIR/stats.json" \
  --out "$OUT_DIR/figs"

echo "Done ($N patients). See $OUT_DIR/figs/ and $OUT_DIR/stats.json"
