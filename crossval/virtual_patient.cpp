#include "virtual_patient.hpp"

#include "uva_padova/config_loader.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/integrals.hpp"
#include "uva_padova/delivery.hpp"
#include "uva_padova/constants.hpp"

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

    if (desc[i].name == "kp1")  p.kp1  *= mult;
    else if (desc[i].name == "kp2")  p.kp2  *= mult;
    else if (desc[i].name == "kp3")  p.kp3  *= mult;
    else if (desc[i].name == "Vm0")  p.Vm0  *= mult;
    else if (desc[i].name == "Vmx")  p.Vmx  *= mult;
    else if (desc[i].name == "p2U")  p.p2U  *= mult;
    else if (desc[i].name == "ka1")  p.ka1  *= mult;
    else if (desc[i].name == "ka2")  p.ka2  *= mult;
    else if (desc[i].name == "kd")   p.kd   *= mult;
    else if (desc[i].name == "kmax") p.kmax *= mult;
    else if (desc[i].name == "kmin") p.kmin *= mult;
    else if (desc[i].name == "kabs") p.kabs *= mult;
    else if (desc[i].name == "kgri") p.kgri *= mult;
    else if (desc[i].name == "VG")   p.VG   *= mult;
    else if (desc[i].name == "Gb")   p.Gb   *= mult;
    else if (desc[i].name == "k1")   p.k1   *= mult;
    else if (desc[i].name == "k2")   p.k2   *= mult;
    else if (desc[i].name == "m1")   p.m1   *= mult;
    else if (desc[i].name == "m2")   p.m2   *= mult;
    else if (desc[i].name == "m4")   p.m4   *= mult;
    else if (desc[i].name == "ki")   p.ki   *= mult;
    else if (desc[i].name == "f")    p.f    *= mult;
    else if (desc[i].name == "ke1")  p.ke1  *= mult;
    else if (desc[i].name == "ke2")  p.ke2  *= mult;
    else if (desc[i].name == "Ts")   p.Ts   *= mult;
    else if (desc[i].name == "r1")   p.r1   *= mult;
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

  // ---- 6. Find steady state for true params ----
  State ss;
  double insulin_ss = SteadyStateSolver::find(ss, result.true_params,
                                               result.true_params.Gb);

  // ---- 7. Build TRUE meal/bolus schedule ----
  struct MealEvent { int t_min; double cho_mg; };
  struct BolusEvent { int t_min; double dose_pmol_kg; };

  std::vector<MealEvent> true_meals;
  std::vector<BolusEvent> true_boluses;

  int n_days = config.duration_min / 1440 + 1;
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

  // ---- 8. Generate REPORTED meals with errors ----
  // reported meals may differ from true meals in timing, amount, or be skipped
  struct ReportedMeal { int t_min; double cho_mg; bool skipped; };
  std::vector<ReportedMeal> reported_meals;
  std::vector<int> true_to_reported_idx; // maps true meal index -> reported index (-1 if skipped)

  std::normal_distribution<double> timing_shift(0.0, config.meal_errors.timing_shift_sd);
  std::normal_distribution<double> amount_noise(0.0, config.meal_errors.amount_noise_sd);
  std::uniform_real_distribution<double> uniform01(0.0, 1.0);

  true_to_reported_idx.reserve(true_meals.size());
  for (const auto& tm : true_meals) {
    // Skip?
    if (config.meal_errors.skip_prob > 0.0 && uniform01(rng) < config.meal_errors.skip_prob) {
      true_to_reported_idx.push_back(-1);
      continue;
    }

    // Timing shift, rounded to nearest minute
    int shift_min = static_cast<int>(std::round(timing_shift(rng)));
    int reported_t = std::max(0, tm.t_min + shift_min);

    // Amount noise (multiplicative, clamped to >= 0)
    double noise_factor = 1.0 + amount_noise(rng);
    if (noise_factor < 0.0) noise_factor = 0.0;
    double reported_cho = tm.cho_mg * noise_factor;

    true_to_reported_idx.push_back(static_cast<int>(reported_meals.size()));
    reported_meals.push_back({reported_t, reported_cho, false});
  }

  // ---- 9. Forward simulation ----
  // We simulate once, recording TRUE CGM and building NOISY CGM simultaneously.
  State x = ss;
  BasicInputs<double> u;
  u.insulin_rate = insulin_ss;

  // True (noiseless) CGM record
  std::vector<double> clean_cgm_time, clean_cgm_value;

  // Noisy CGM record (with noise + dropouts)
  std::vector<double> noisy_cgm_time, noisy_cgm_value;

  // Bolus record (same for both)
  std::vector<double> bolus_time_vec, bolus_dose_vec;

  // CGM dropout state
  bool dropout_active = false;
  int dropout_remaining = 0;

  std::normal_distribution<double> cgm_noise_unit(0.0, 1.0);

  // Convert dropout duration from minutes to CGM observation intervals
  int max_dropout_obs = config.cgm_noise.max_dropout_min / config.cgm_interval;
  if (max_dropout_obs < 1) max_dropout_obs = 1;
  std::uniform_int_distribution<int> dropout_duration_dist(1, max_dropout_obs);

  int meal_idx = 0, bolus_idx = 0;

  for (int t = 0; t <= config.duration_min; ++t) {
    // Apply TRUE meal events (physiological reality)
    while (meal_idx < static_cast<int>(true_meals.size()) && true_meals[meal_idx].t_min == t) {
      x.Qsto1 += true_meals[meal_idx].cho_mg;
      x.mealD = true_meals[meal_idx].cho_mg;
      meal_idx++;
    }

    // Apply TRUE bolus events
    while (bolus_idx < static_cast<int>(true_boluses.size()) && true_boluses[bolus_idx].t_min == t) {
      x.Isc1 += true_boluses[bolus_idx].dose_pmol_kg;
      bolus_time_vec.push_back(static_cast<double>(t));
      bolus_dose_vec.push_back(true_boluses[bolus_idx].dose_pmol_kg);
      bolus_idx++;
    }

    // Record CGM at specified interval
    if (t % config.cgm_interval == 0) {
      double true_gsc = x.Gsc;

      // Clean CGM (noiseless)
      clean_cgm_time.push_back(static_cast<double>(t));
      clean_cgm_value.push_back(true_gsc);

      // Noisy CGM with dropouts
      noisy_cgm_time.push_back(static_cast<double>(t));

      // Start a new dropout if not in one
      if (!dropout_active &&
          config.cgm_noise.max_dropout_min > 0 &&
          config.cgm_noise.dropout_prob > 0.0 &&
          uniform01(rng) < config.cgm_noise.dropout_prob) {
        dropout_active = true;
        dropout_remaining = dropout_duration_dist(rng);
      }

      if (dropout_active && dropout_remaining > 0) {
        // Convention: 0 = missing CGM
        noisy_cgm_value.push_back(0.0);
        dropout_remaining--;
        if (dropout_remaining <= 0) {
          dropout_active = false;
        }
      } else {
        // Heteroskedastic noise: σ_total = sqrt(σ_fixed² + (σ_rel · Gsc / 100)²)
        double sigma_fixed = config.cgm_noise.fixed_sd;
        double sigma_rel = (config.cgm_noise.relative_sd / 100.0) * true_gsc;
        double sigma = std::sqrt(sigma_fixed * sigma_fixed + sigma_rel * sigma_rel);
        double obs = true_gsc + cgm_noise_unit(rng) * sigma;
        if (obs <= 0.0) obs = 0.1;  // avoid confusion with dropout flag
        noisy_cgm_value.push_back(obs);
      }
    }

    // RK4 step
    if (t < config.duration_min) {
      RK4Integrator::step(UvaPadovaODESystem{}, x, u, result.true_params, 1.0);
    }
  }

  // ---- 10. Build CLEAN IdentificationData (noiseless CGM, true meals) ----
  {
    IdentificationData clean;
    clean.cgm_time = clean_cgm_time;
    clean.cgm_value = clean_cgm_value;
    for (const auto& m : true_meals) {
      clean.meal_time.push_back(static_cast<double>(m.t_min));
      clean.meal_cho.push_back(m.cho_mg);
    }
    clean.bolus_time = bolus_time_vec;
    clean.bolus_dose = bolus_dose_vec;
    clean.finalize();
    result.true_data = std::move(clean);
  }

  // ---- 11. Build NOISY IdentificationData (noisy CGM, reported meals) ----
  {
    IdentificationData noisy;
    // CGM: skip values that are 0 (missing/dropout convention)
    for (size_t i = 0; i < noisy_cgm_time.size(); ++i) {
      if (noisy_cgm_value[i] > 0.0) {
        noisy.cgm_time.push_back(noisy_cgm_time[i]);
        noisy.cgm_value.push_back(noisy_cgm_value[i]);
      }
    }

    // Meals: use reported meals
    for (const auto& rm : reported_meals) {
      noisy.meal_time.push_back(static_cast<double>(rm.t_min));
      noisy.meal_cho.push_back(rm.cho_mg);
    }

    // Boluses: same as true (pump records accurate delivery)
    noisy.bolus_time = bolus_time_vec;
    noisy.bolus_dose = bolus_dose_vec;

    noisy.finalize();
    result.data = std::move(noisy);
  }

  return result;
}

} // namespace crossval
} // namespace uva_padova
