#!/usr/bin/env bash
# Run multi-day experiment: for 1,2,3,5,10 days, run the pipeline and
# collect gate results.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_DIR/build"
OUTPUT_DIR="$REPO_DIR/validation/output"
TEST_BIN="$BUILD_DIR/validation/test_real_data"
PARSER="$REPO_DIR/validation/parse_patient_data.py"

# Configs: (days, cmaes_evals)
CONFIGS=(
  "1 500"
  "2 500"
  "3 300"
  "5 200"
  "10 150"
)

RESULTS_FILE="$OUTPUT_DIR/multi_day_results.txt"
echo "# Multi-day gate experiment" > "$RESULTS_FILE"
echo "# days, CGM, meals, boluses, Gb, cost, gate_a_clusters, gate_a_passed, gate_b_passed, ljung_q, ljung_p, regime_bias, gate_c_prior_dominated, total_time_s" >> "$RESULTS_FILE"

for config in "${CONFIGS[@]}"; do
  read -r ndays evals <<< "$config"
  suffix="${ndays}d"

  echo ""
  echo "========================================="
  echo "  Running: ${ndays} day(s), ${evals} evals"
  echo "========================================="

  CGM_CSV="$OUTPUT_DIR/patient563_${suffix}_cgm_meals_bolus.csv"
  BASAL_CSV="$OUTPUT_DIR/patient563_${suffix}_basal.csv"

  start=$(date +%s)

  # Run pipeline
  output=$("$TEST_BIN" "$CGM_CSV" "$BASAL_CSV" 3 "$evals" 2>&1)
  echo "$output"

  end=$(date +%s)
  elapsed=$((end - start))

  # Parse results
  cgm_count=$(echo "$output" | grep "loaded" | tail -1 | sed -E 's/.*loaded ([0-9]+) CGM.*/\1/')
  meal_count=$(echo "$output" | grep "loaded" | tail -1 | sed -E 's/.* ([0-9]+) meals.*/\1/')
  bolus_count=$(echo "$output" | grep "loaded" | tail -1 | sed -E 's/.* ([0-9]+) boluses.*/\1/')
  gb=$(echo "$output" | grep "Estimated Gb" | sed -E 's/.*Gb = ([0-9.]+).*/\1/')
  cost=$(echo "$output" | grep "^  cost=" | sed -E 's/.*cost=([0-9.eE+-]+).*/\1/')

  gate_a_clusters=$(echo "$output" | grep "clusters=" | sed -E 's/.*clusters=([0-9]+).*/\1/')
  gate_a_passed=$(echo "$output" | grep "passed=" | sed -E 's/.*passed=(yes|NO).*/\1/')

  gate_b_passed=$(echo "$output" | grep "No structural issues\|FAILED" | sed -E 's/.*(No structural|FAILED).*/\1/' | head -1)
  ljung_q=$(echo "$output" | grep "Ljung-Box:" | sed -E 's/.*Q=([0-9.eE+-]+).*/\1/')
  ljung_p=$(echo "$output" | grep "Ljung-Box:" | sed -E 's/.*p=([0-9.eE+-]+).*/\1/')
  regime_bias=$(echo "$output" | grep "Regime bias detected:" | sed -E 's/.*detected: (yes|no).*/\1/')

  gate_c_labels=$(echo "$output" | grep "Per-param labels:" | sed 's/.*Per-param labels://' | xargs)
  n_prior=$(echo "$gate_c_labels" | grep -o "prior-dominated" | wc -l | xargs)

  echo "$ndays, $cgm_count, $meal_count, $bolus_count, $gb, $cost, $gate_a_clusters, $gate_a_passed, $gate_b_passed, $ljung_q, $ljung_p, $regime_bias, $n_prior, $elapsed" >> "$RESULTS_FILE"

done

echo ""
echo "========================================="
echo "  RESULTS"
echo "========================================="
column -t -s',' "$RESULTS_FILE"
