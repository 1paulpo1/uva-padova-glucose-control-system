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

    // Base meal times: from config or auto-spaced 360..1260
    std::vector<double> base_times = mg.meal_times;
    if (base_times.empty() && mg.n_meals_per_day > 0) {
      if (mg.n_meals_per_day == 1) {
        base_times.push_back(720.0);
      } else {
        for (int i = 0; i < mg.n_meals_per_day; ++i)
          base_times.push_back(360.0 + i * 900.0 / (mg.n_meals_per_day - 1));
      }
    }

    for (int day = 0; day < n_days; ++day) {
      int day_offset = day * 1440;

      // Per-day RNG (disjoint from theta: *10000, from syn_data: day*7)
      std::mt19937 day_rng(config.seed + patient_id * 10000 + day * 7);
      std::lognormal_distribution<double> ln_cho(0.0, mg.split_cv);
      std::lognormal_distribution<double> ln_time(0.0, mg.time_cv);

      int n = mg.n_meals_per_day;

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
