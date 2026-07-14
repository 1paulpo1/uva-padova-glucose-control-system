#pragma once

#include "uva_padova/cost_function.hpp"
#include <random>

namespace uva_padova {

// ---------------------------------------------------------------------------
// Configuration for synthetic data generation
// ---------------------------------------------------------------------------
struct SyntheticDataConfig {
  int duration_min{720};                         // total simulation time [min]
  int cgm_interval{5};                           // CGM sampling interval [min]
  double noise_sd{3.0};                          // std dev of additive Gaussian noise [mg/dL]
  int seed{42};
  std::vector<std::pair<double, double>> meals;   // (time_min, cho_mg)
  std::vector<std::pair<double, double>> boluses; // (time_min, dose_pmol_kg)
  std::vector<std::pair<double, double>> insulin; // (time_min, rate_pmol_kg_per_min)

  // Richer noise model
  double relative_sd{0.0};        // %-based: σ += (pct/100) * Gsc
  double dropout_prob{0.0};       // probability of starting a dropout at any observation
  int max_dropout_min{0};         // max dropout duration [min] (0 = none)

  // Meal errors
  double meal_skip_prob{0.0};         // probability of skipping a meal entirely
  double meal_timing_shift_sd{0.0};   // std dev of time reporting error [min]
  double meal_amount_noise_sd{0.0};   // std dev of CHO fraction (0.15 = 15%)

  // Insulin errors (bolus)
  double bolus_skip_prob{0.0};        // probability of skipping a bolus
  double bolus_dose_cv{0.0};          // LogNormal CV for bolus dose error
  double bolus_timing_shift_sd{0.0};  // std dev of bolus timing error [min]

  // Insulin errors (basal)
  double basal_rate_cv{0.0};          // LogNormal CV for basal rate error
};

// ---------------------------------------------------------------------------
// Result of synthetic data generation
// ---------------------------------------------------------------------------
struct SyntheticDataResult {
  IdentificationData data;       // noisy CGM + reported meals (if errors applied)
  IdentificationData true_data;  // clean (noiseless) CGM + true meals
};

// ---------------------------------------------------------------------------
// Compute log-space theta from true and base params
//   theta[i] = log(true_params.field_i / base_params.field_i)
// ---------------------------------------------------------------------------
inline std::vector<double> compute_theta(
    const Params& true_params,
    const Params& base_params,
    const std::vector<ParamDescriptor>& desc)
{
  std::vector<double> theta(desc.size());
  for (size_t i = 0; i < desc.size(); ++i) {
    double base = param_value(base_params, desc[i].name);
    double true_val = param_value(true_params, desc[i].name);
    theta[i] = std::log(true_val / base);
  }
  return theta;
}

// ---------------------------------------------------------------------------
// Generate synthetic CGM data for a given set of true params.
//
// Steps:
//   1. Compute theta_true = log(true / base) for each active param
//   2. Build IdentificationData scaffold (observation times, meals, etc.)
//   3. Simulate clean CGM via ObservationCost::simulate_observations
//   4. Add noise and optionally apply error models
//   5. Return SyntheticDataResult with both clean and noisy data
// ---------------------------------------------------------------------------
template <typename Simulator = UvaPadovaSimulator>
SyntheticDataResult generate_synthetic_data(
    const Params& true_params,
    const Params& base_params,
    const std::vector<ParamDescriptor>& desc,
    const SyntheticDataConfig& config)
{
  auto theta = compute_theta(true_params, base_params, desc);

  // ---- Build scaffold with TRUE meals ----
  IdentificationData scaffold;
  for (int t = 0; t <= config.duration_min; t += config.cgm_interval)
    scaffold.cgm_time.push_back(static_cast<double>(t));
  scaffold.cgm_value.assign(scaffold.cgm_time.size(), 0.0);

  for (const auto& m : config.meals) {
    scaffold.meal_time.push_back(m.first);
    scaffold.meal_cho.push_back(m.second);
  }
  for (const auto& b : config.boluses) {
    scaffold.bolus_time.push_back(b.first);
    scaffold.bolus_dose.push_back(b.second);
  }
  for (const auto& i : config.insulin) {
    scaffold.insulin_time.push_back(i.first);
    scaffold.insulin_rate.push_back(i.second);
  }
  scaffold.finalize();

  // ---- Simulate clean CGM ----
  ObservationCost<Simulator> cost(scaffold, base_params, desc);
  auto clean_cgm = cost.template simulate_observations<double>(theta.data());

  std::mt19937 rng(config.seed);
  std::normal_distribution<double> std_normal(0.0, 1.0);

  // ---- Determine if rich error model is requested ----
  bool rich_noise = (config.relative_sd > 0.0 ||
                     config.dropout_prob > 0.0 ||
                     config.meal_skip_prob > 0.0 ||
                     config.meal_timing_shift_sd > 0.0 ||
                     config.meal_amount_noise_sd > 0.0 ||
                     config.bolus_skip_prob > 0.0 ||
                     config.bolus_dose_cv > 0.0 ||
                     config.bolus_timing_shift_sd > 0.0 ||
                     config.basal_rate_cv > 0.0);

  if (!rich_noise) {
    // Simple i.i.d. Gaussian noise (original behavior)
    SyntheticDataResult result;

    // Noisy data
    result.data.cgm_time = scaffold.cgm_time;
    result.data.cgm_value.resize(clean_cgm.size());
    for (size_t i = 0; i < clean_cgm.size(); ++i)
      result.data.cgm_value[i] = clean_cgm[i] + std_normal(rng) * config.noise_sd;
    result.data.meal_time = scaffold.meal_time;
    result.data.meal_cho = scaffold.meal_cho;
    result.data.bolus_time = scaffold.bolus_time;
    result.data.bolus_dose = scaffold.bolus_dose;
    result.data.insulin_time = scaffold.insulin_time;
    result.data.insulin_rate = scaffold.insulin_rate;
    result.data.finalize();

    // Clean data
    result.true_data.cgm_time = scaffold.cgm_time;
    result.true_data.cgm_value = clean_cgm;
    result.true_data.meal_time = scaffold.meal_time;
    result.true_data.meal_cho = scaffold.meal_cho;
    result.true_data.bolus_time = scaffold.bolus_time;
    result.true_data.bolus_dose = scaffold.bolus_dose;
    result.true_data.insulin_time = scaffold.insulin_time;
    result.true_data.insulin_rate = scaffold.insulin_rate;
    result.true_data.finalize();

    return result;
  }

  // ---- Rich error model (crossval-style) ----

  // 1. Generate reported meals with errors
  std::normal_distribution<double> timing_shift(0.0, config.meal_timing_shift_sd);
  std::normal_distribution<double> amount_noise(0.0, config.meal_amount_noise_sd);
  std::uniform_real_distribution<double> uniform01(0.0, 1.0);

  std::vector<std::pair<double, double>> reported_meals;
  for (size_t i = 0; i < config.meals.size(); ++i) {
    // Skip?
    if (config.meal_skip_prob > 0.0 && uniform01(rng) < config.meal_skip_prob)
      continue;

    // Timing shift, rounded to nearest minute
    int shift = static_cast<int>(std::round(timing_shift(rng)));
    double reported_t = std::max(0.0, config.meals[i].first + static_cast<double>(shift));

    // Amount noise (multiplicative, clamped to >= 0)
    double noise_factor = 1.0 + amount_noise(rng);
    if (noise_factor < 0.0) noise_factor = 0.0;
    double reported_cho = config.meals[i].second * noise_factor;

    reported_meals.emplace_back(reported_t, reported_cho);
  }

  // 1b. Apply errors to boluses
  std::lognormal_distribution<double> bolus_dose_noise(0.0, config.bolus_dose_cv);
  std::normal_distribution<double> bolus_timing_shift(0.0, config.bolus_timing_shift_sd);

  std::vector<std::pair<double, double>> reported_boluses;
  for (size_t i = 0; i < config.boluses.size(); ++i) {
    if (config.bolus_skip_prob > 0.0 && uniform01(rng) < config.bolus_skip_prob)
      continue;

    int shift = static_cast<int>(std::round(bolus_timing_shift(rng)));
    double reported_t = std::max(0.0, config.boluses[i].first + static_cast<double>(shift));

    double dose_factor = (config.bolus_dose_cv > 0.0) ? bolus_dose_noise(rng) : 1.0;
    double reported_dose = config.boluses[i].second * dose_factor;

    reported_boluses.emplace_back(reported_t, reported_dose);
  }

  // 1c. Apply errors to basal rate
  std::lognormal_distribution<double> basal_noise(0.0, config.basal_rate_cv);
  std::vector<std::pair<double, double>> reported_insulin;
  for (const auto& ins : config.insulin) {
    double rate_factor = (config.basal_rate_cv > 0.0) ? basal_noise(rng) : 1.0;
    reported_insulin.emplace_back(ins.first, ins.second * rate_factor);
  }

  // 2. Apply CGM noise + dropouts
  std::vector<double> noisy_cgm_time, noisy_cgm_value;

  // Dropout state
  bool dropout_active = false;
  int dropout_remaining = 0;
  int max_dropout_obs = config.max_dropout_min / config.cgm_interval;
  if (max_dropout_obs < 1) max_dropout_obs = 1;
  std::uniform_int_distribution<int> dropout_duration_dist(1, max_dropout_obs);

  for (size_t i = 0; i < clean_cgm.size(); ++i) {
    double t = scaffold.cgm_time[i];
    double true_gsc = clean_cgm[i];

    // Start a new dropout if not in one
    if (!dropout_active &&
        config.max_dropout_min > 0 &&
        config.dropout_prob > 0.0 &&
        uniform01(rng) < config.dropout_prob) {
      dropout_active = true;
      dropout_remaining = dropout_duration_dist(rng);
    }

    if (dropout_active && dropout_remaining > 0) {
      // Convention: 0 = missing CGM
      noisy_cgm_time.push_back(t);
      noisy_cgm_value.push_back(0.0);
      dropout_remaining--;
      if (dropout_remaining <= 0)
        dropout_active = false;
    } else {
      // Heteroskedastic noise: σ = √(σ_fixed² + (σ_rel/100 · Gsc)²)
      double sigma_fixed = config.noise_sd;
      double sigma_rel = (config.relative_sd / 100.0) * true_gsc;
      double sigma = std::sqrt(sigma_fixed * sigma_fixed + sigma_rel * sigma_rel);
      double obs = true_gsc + std_normal(rng) * sigma;
      if (obs <= 0.0) obs = 0.1;  // avoid confusion with dropout flag
      noisy_cgm_time.push_back(t);
      noisy_cgm_value.push_back(obs);
    }
  }

  // 3. Build result
  SyntheticDataResult result;

  // Noisy data: filter dropouts (0 values are missing), use reported meals
  for (size_t i = 0; i < noisy_cgm_time.size(); ++i) {
    if (noisy_cgm_value[i] > 0.0) {
      result.data.cgm_time.push_back(noisy_cgm_time[i]);
      result.data.cgm_value.push_back(noisy_cgm_value[i]);
    }
  }
  for (const auto& rm : reported_meals) {
    result.data.meal_time.push_back(rm.first);
    result.data.meal_cho.push_back(rm.second);
  }
  for (const auto& rb : reported_boluses) {
    result.data.bolus_time.push_back(rb.first);
    result.data.bolus_dose.push_back(rb.second);
  }
  for (const auto& ri : reported_insulin) {
    result.data.insulin_time.push_back(ri.first);
    result.data.insulin_rate.push_back(ri.second);
  }
  result.data.finalize();

  // Clean data: noiseless CGM, true meals
  result.true_data.cgm_time = scaffold.cgm_time;
  result.true_data.cgm_value = clean_cgm;
  result.true_data.meal_time = scaffold.meal_time;
  result.true_data.meal_cho = scaffold.meal_cho;
  result.true_data.bolus_time = scaffold.bolus_time;
  result.true_data.bolus_dose = scaffold.bolus_dose;
  result.true_data.insulin_time = scaffold.insulin_time;
  result.true_data.insulin_rate = scaffold.insulin_rate;
  result.true_data.finalize();

  return result;
}

} // namespace uva_padova
