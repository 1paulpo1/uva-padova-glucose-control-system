#pragma once

#include <vector>
#include <string>

#include "uva_padova/types.hpp"
#include "uva_padova/cost_function.hpp"
#include "uva_padova/constants.hpp"

namespace uva_padova {
namespace crossval {

// ---------------------------------------------------------------------------
// CGM noise configuration
// ---------------------------------------------------------------------------
struct CGMNoiseConfig {
  double fixed_sd{3.0};        // fixed std dev [mg/dL]
  double relative_sd{0.0};     // percentage-based: σ += (pct/100) * Gsc
  int max_dropout_min{0};      // max CGM dropout duration [min] (0 = none)
  double dropout_prob{0.0};    // prob of starting a dropout at any observation
};

// ---------------------------------------------------------------------------
// Insulin error configuration (applied to both bolus and basal)
// ---------------------------------------------------------------------------
struct InsulinErrorConfig {
  double bolus_skip_prob{0.0};        // probability of skipping a bolus
  double bolus_dose_cv{0.0};          // LogNormal CV for bolus dose error
  double bolus_timing_shift_sd{0.0};  // std dev of bolus timing error [min]
  double basal_rate_cv{0.0};          // LogNormal CV for basal rate error
};

// ---------------------------------------------------------------------------
// Meal error configuration
// ---------------------------------------------------------------------------
struct MealErrorConfig {
  double skip_prob{0.0};        // prob of skipping a meal entirely
  double timing_shift_sd{0.0};  // std dev of time reporting error [min]
  double amount_noise_sd{0.0};  // std dev of CHO fraction (0.1 = 10%)
};

// ---------------------------------------------------------------------------
// Generative meal configuration (replaces static daily_meals when enabled)
// ---------------------------------------------------------------------------
struct MealGenerationConfig {
  double daily_cho_g{210.0};    // total CHO per day [g]
  int n_meals_per_day{3};       // number of meals per day
  double split_cv{0.3};         // LogNormal CV for CHO split across meals
  double time_cv{0.12};         // LogNormal CV for meal time jitter (0 = fixed)
  std::vector<double> meal_times; // base time of each meal [min]; empty = auto-spaced
  double bolus_u_per_g{0.1};    // insulin units per gram CHO
  bool enabled{false};
};

// ---------------------------------------------------------------------------
// Configuration for the cross-validation experiment
// ---------------------------------------------------------------------------
struct VirtualPatientConfig {
  int duration_min{4320};        // 3 days
  int cgm_interval{5};           // CGM sampling interval [min]
  int seed{42};                  // base seed (patient_id added)
  int n_cmaes_restarts{3};
  int cmaes_evals{1000};
  double perturbation_std{0.3};  // std dev of θ in log-space
  std::string params_config_path;  // path to JSON param descriptors (empty = default)

  CGMNoiseConfig cgm_noise;
  InsulinErrorConfig insulin_errors;
  MealErrorConfig meal_errors;
  MealGenerationConfig meal_generation;

  double body_weight_kg{75.0};   // default body weight [kg], used for bolus_u→pmol/kg

  struct DailyMeal {
    int time_min;                // minutes from start of day
    double cho_mg;               // carbohydrate amount [mg]
    double bolus_dose_pmol_kg;   // bolus dose [pmol/kg] (0 = no bolus)
  };
  std::vector<DailyMeal> daily_meals;
};

// Default meal schedule: breakfast 60g, lunch 80g, dinner 70g, with boluses
inline VirtualPatientConfig default_config() {
  VirtualPatientConfig cfg;
  cfg.cgm_noise.fixed_sd = 3.0;
  cfg.daily_meals = {
    { 360, 60000,  6.0 * constants::kInsulinUnits_To_Pmol / cfg.body_weight_kg },
    { 720, 80000,  8.0 * constants::kInsulinUnits_To_Pmol / cfg.body_weight_kg },
    { 1080, 70000, 7.0 * constants::kInsulinUnits_To_Pmol / cfg.body_weight_kg },
  };
  return cfg;
}

// ---------------------------------------------------------------------------
// Result of generating a virtual patient
// ---------------------------------------------------------------------------
struct VirtualPatientResult {
  Params true_params;                      // ground truth parameters
  Params base_params;                      // base (defaults_t1d())
  IdentificationData data;                 // observed data (noise + errors applied)
  IdentificationData true_data;            // clean data (noiseless CGM, correct meals)
  std::vector<ParamDescriptor> all_desc;   // all descriptors from config
  std::vector<ParamDescriptor> active_desc;// only non-fixed (corrected priors)
  std::vector<double> theta_true;          // true θ in log-space (active params only)
  int patient_id{0};
};

// ---------------------------------------------------------------------------
// Generate a virtual patient with perturbed params + configurable noise
// ---------------------------------------------------------------------------
VirtualPatientResult generate_virtual_patient(
    const VirtualPatientConfig& config,
    int patient_id);

// ---------------------------------------------------------------------------
// Apply log-space theta vector to base params, producing true params
// ---------------------------------------------------------------------------
Params apply_theta_to_params(
    const Params& base,
    const std::vector<ParamDescriptor>& desc,
    const std::vector<double>& theta);

} // namespace crossval
} // namespace uva_padova
