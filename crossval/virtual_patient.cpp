#include "virtual_patient.hpp"

#include "uva_padova/config_loader.hpp"
#include "uva_padova/synthetic_data_generator.hpp"

#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace uva_padova {
namespace crossval {

// ---------------------------------------------------------------------------
// Convert (target_mean, target_stddev) → (n_trials, p) for binomial_distribution
// such that Binomial(n, p) has mean ≈ target_mean and std ≈ target_stddev.
// ---------------------------------------------------------------------------
static std::pair<int, double> binomial_params_from_mean_std(double mean, double stddev) {
  if (stddev <= 0.0) {
    return {static_cast<int>(std::round(mean)), 1.0};
  }
  double var = stddev * stddev;
  double max_var = mean * 0.999;
  double eff_var = (var < max_var) ? var : max_var;
  double p = 1.0 - eff_var / mean;
  p = std::clamp(p, 0.001, 0.999);
  int n = static_cast<int>(std::round(mean / p));
  n = std::max(n, static_cast<int>(std::round(mean)));
  return {n, p};
}

// ---------------------------------------------------------------------------
// Apply log-space theta vector to base params, producing true params
// ---------------------------------------------------------------------------
Params apply_theta_to_params(
    const Params& base,
    const std::vector<ParamDescriptor>& desc,
    const std::vector<double>& theta)
{
  Params p = base;
  for (size_t i = 0; i < theta.size() && i < desc.size(); ++i) {
    double mult = std::exp(theta[i]);
    if (!std::isfinite(mult)) mult = 1.0;
    param_by_id(p, param_id_from_string(desc[i].name)) *= mult;
  }
  return p;
}

// ---------------------------------------------------------------------------
// Generate a virtual patient with configurable noise and errors
// ---------------------------------------------------------------------------
VirtualPatientResult generate_virtual_patient(
    const VirtualPatientConfig& config,
    int patient_id)
{
  VirtualPatientResult result;
  result.patient_id = patient_id;

  // ---- 1. Load parameter descriptors ----
  if (!config.params_config_path.empty())
    result.all_desc = load_param_descriptors(config.params_config_path);
  else
    result.all_desc = load_default_param_descriptors();
  if (result.all_desc.empty()) {
    std::cerr << "[virtual_patient] ERROR: failed to load default params\n";
    return result;
  }

  // ---- 2. Filter to only active (non-fixed) params ----
  auto raw_active = active_descriptors(result.all_desc);
  result.active_desc = raw_active;

  // ---- 3. Fix prior_mean bug: should be 0.0 in log-space ----
  for (auto& d : result.active_desc) {
    d.prior_mean = 0.0;
    d.initial_val = 0.0;
  }

  // ---- 4. Generate true θ in log-space for active params ----
  int n_active = static_cast<int>(result.active_desc.size());
  int rng_seed = config.seed + patient_id * 1000;
  std::mt19937 rng(rng_seed);

  std::normal_distribution<double> theta_dist(0.0, config.perturbation_std);
  result.theta_true.resize(n_active);
  for (int i = 0; i < n_active; ++i) {
    result.theta_true[i] = theta_dist(rng);
  }

  // ---- 5. Build ground truth params ----
  result.base_params = Params::defaults_t1d();
  result.true_params = apply_theta_to_params(
      result.base_params, result.active_desc, result.theta_true);

  // ---- 6. Build TRUE meal/bolus schedule ----
  struct MealEvent { int t_min; double cho_mg; };
  struct BolusEvent { int t_min; double dose_pmol_kg; };

  std::vector<MealEvent> true_meals;
  std::vector<BolusEvent> true_boluses;

  int n_days = config.duration_min / 1440 + 1;

  if (config.meal_generation.enabled) {
    // ---- Generative meal schedule: different every day ----
    const auto& mg = config.meal_generation;

    // ---- Level 1: Per-patient base meal count ----
    int patient_base_n = mg.n_meals_per_day;
    if (mg.n_meals_mean > 0.0) {
      auto [n_p, p_p] = binomial_params_from_mean_std(
          mg.n_meals_mean, mg.n_meals_std_patient);
      std::mt19937 patient_rng(config.seed + patient_id * 100000 + 1234);
      std::binomial_distribution<int> patient_dist(n_p, p_p);
      patient_base_n = std::max(1, patient_dist(patient_rng));
    }

    for (int day = 0; day < n_days; ++day) {
      int day_offset = day * 1440;

      // Per-day RNG (disjoint from theta: *10000, from syn_data: day*7)
      std::mt19937 day_rng(config.seed + patient_id * 10000 + day * 7);
      std::lognormal_distribution<double> ln_cho(0.0, mg.split_cv);
      std::lognormal_distribution<double> ln_time(0.0, mg.time_cv);

      // ---- Level 2: Per-day meal count ----
      int n = patient_base_n;
      if (mg.n_meals_mean > 0.0 && mg.n_meals_std_day > 0.0) {
        auto [n_d, p_d] = binomial_params_from_mean_std(
            static_cast<double>(patient_base_n), mg.n_meals_std_day);
        std::binomial_distribution<int> day_dist(n_d, p_d);
        n = std::max(1, day_dist(day_rng));
      }

      // Base meal times: from config or auto-spaced 360..1260
      std::vector<double> base_times = mg.meal_times;
      if (base_times.empty() && n > 0) {
        if (n == 1) {
          base_times.push_back(720.0);
        } else {
          for (int i = 0; i < n; ++i)
            base_times.push_back(360.0 + i * 900.0 / (n - 1));
        }
      }

      if (n == 0) continue;

      // Raw CHO fractions
      std::vector<double> raw(n);
      double sum = 0.0;
      for (int i = 0; i < n; ++i) {
        raw[i] = ln_cho(day_rng);
        sum += raw[i];
      }

      for (int i = 0; i < n; ++i) {
        double fraction = raw[i] / sum;
        double cho_mg = mg.daily_cho_g * fraction * 1000.0;

        // Time with LogNormal jitter
        double t_raw = base_times[i] * ln_time(day_rng);
        t_raw = std::clamp(t_raw, 30.0, 1410.0);
        int t_min = day_offset + static_cast<int>(std::round(t_raw));
        if (t_min > config.duration_min) break;

        // Ensure ≥30 min gap from previous meal
        if (!true_meals.empty() && t_min <= true_meals.back().t_min + 30)
          t_min = true_meals.back().t_min + 30;

        // Bolus dose
        double bolus_pmol_kg = (cho_mg / 1000.0) * mg.bolus_u_per_g
                             * constants::kInsulinUnits_To_Pmol / config.body_weight_kg;

        true_meals.push_back({t_min, cho_mg});
        if (bolus_pmol_kg > 0.0)
          true_boluses.push_back({t_min, bolus_pmol_kg});
      }
    }
  } else {
    // ---- Static daily_meals template (original behavior) ----
    for (int day = 0; day < n_days; ++day) {
      int day_offset = day * 1440;
      for (const auto& dm : config.daily_meals) {
        int t = day_offset + dm.time_min;
        if (t > config.duration_min) break;
        true_meals.push_back({t, dm.cho_mg});
        if (dm.bolus_dose_pmol_kg > 0.0) {
          true_boluses.push_back({t, dm.bolus_dose_pmol_kg});
        }
      }
    }
  }

  // ---- 7. Correction boluses (insulin without meals) ----
  const auto& cb = config.correction_bolus;
  if (cb.prob_per_hour > 0.0) {
    std::mt19937 cb_rng(config.seed + patient_id * 100000 + 5678);

    double duration_hours = config.duration_min / 60.0;
    std::poisson_distribution<int> n_correction_dist(cb.prob_per_hour * duration_hours);
    int n_correction = n_correction_dist(cb_rng);

    if (n_correction > 0) {
      std::gamma_distribution<double> dose_dist(
          (cb.dose_mean_U * cb.dose_mean_U) / (cb.dose_std_U * cb.dose_std_U),
          (cb.dose_std_U * cb.dose_std_U) / cb.dose_mean_U);
      std::uniform_real_distribution<double> time_dist(0.0, static_cast<double>(config.duration_min));

      for (int i = 0; i < n_correction; ++i) {
        int t_min = static_cast<int>(std::round(time_dist(cb_rng)));
        double dose_U = dose_dist(cb_rng);
        double dose_pmol_kg = dose_U
            * constants::kInsulinUnits_To_Pmol / config.body_weight_kg;
        true_boluses.emplace_back(t_min, dose_pmol_kg);
      }

      std::sort(true_boluses.begin(), true_boluses.end(),
          [](const BolusEvent& a, const BolusEvent& b) { return a.t_min < b.t_min; });
    }
  }

  // ---- 8. Generate synthetic data via shared generator ----
  // The generator handles: forward simulation (via simulate_observations),
  // heteroskedastic CGM noise, dropouts, meal errors, and building both
  // clean and noisy IdentificationData.
  SyntheticDataConfig gen_cfg;
  gen_cfg.duration_min = config.duration_min;
  gen_cfg.cgm_interval = config.cgm_interval;
  gen_cfg.noise_sd = config.cgm_noise.fixed_sd;
  gen_cfg.seed = rng_seed + 1;  // +1 to avoid overlap with theta generation RNG
  gen_cfg.relative_sd = config.cgm_noise.relative_sd;
  gen_cfg.dropout_prob = config.cgm_noise.dropout_prob;
  gen_cfg.max_dropout_min = config.cgm_noise.max_dropout_min;
  gen_cfg.meal_skip_prob = config.meal_errors.skip_prob;
  gen_cfg.meal_timing_shift_sd = config.meal_errors.timing_shift_sd;
  gen_cfg.meal_amount_noise_sd = config.meal_errors.amount_noise_sd;

  gen_cfg.bolus_skip_prob = config.insulin_errors.bolus_skip_prob;
  gen_cfg.bolus_dose_cv = config.insulin_errors.bolus_dose_cv;
  gen_cfg.bolus_timing_shift_sd = config.insulin_errors.bolus_timing_shift_sd;
  gen_cfg.basal_rate_cv = config.insulin_errors.basal_rate_cv;

  for (const auto& m : true_meals)
    gen_cfg.meals.emplace_back(static_cast<double>(m.t_min), m.cho_mg);
  for (const auto& b : true_boluses)
    gen_cfg.boluses.emplace_back(static_cast<double>(b.t_min), b.dose_pmol_kg);

  auto gen_result = generate_synthetic_data(
      result.true_params, result.base_params, result.active_desc, gen_cfg);
  result.true_data = std::move(gen_result.true_data);
  result.data = std::move(gen_result.data);

  return result;
}

} // namespace crossval
} // namespace uva_padova
