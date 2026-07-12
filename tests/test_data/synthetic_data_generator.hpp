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
};

// ---------------------------------------------------------------------------
// Extract a parameter value from Params by name
// ---------------------------------------------------------------------------
inline double param_value(const Params& p, const std::string& name) {
  if (name == "VG")   return p.VG;   if (name == "k1")   return p.k1;   if (name == "k2")   return p.k2;
  if (name == "VI")   return p.VI;   if (name == "m1")   return p.m1;   if (name == "m2")   return p.m2;
  if (name == "m4")   return p.m4;   if (name == "kmax") return p.kmax; if (name == "kmin") return p.kmin;
  if (name == "kabs") return p.kabs; if (name == "kgri") return p.kgri; if (name == "f")    return p.f;
  if (name == "b")    return p.b;    if (name == "d")    return p.d;
  if (name == "kp1")  return p.kp1;  if (name == "kp2")  return p.kp2;  if (name == "kp3")  return p.kp3;
  if (name == "Vm0")  return p.Vm0;  if (name == "Vmx")  return p.Vmx;  if (name == "p2U")  return p.p2U;
  if (name == "ka1")  return p.ka1;  if (name == "ka2")  return p.ka2;  if (name == "kd")   return p.kd;
  if (name == "Fcns") return p.Fcns; if (name == "Km0")  return p.Km0;
  if (name == "r1")   return p.r1;   if (name == "Gth")  return p.Gth;
  if (name == "ke1")  return p.ke1;  if (name == "ke2")  return p.ke2;
  if (name == "Ts")   return p.Ts;   if (name == "BW")   return p.BW;
  if (name == "Gb")   return p.Gb;   if (name == "Ib")   return p.Ib;
  return 1.0;
}

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
//   4. Add i.i.d. Gaussian noise
//   5. Return IdentificationData ready for pipeline identification
// ---------------------------------------------------------------------------
template <typename Simulator = UvaPadovaSimulator>
IdentificationData generate_synthetic_data(
    const Params& true_params,
    const Params& base_params,
    const std::vector<ParamDescriptor>& desc,
    const SyntheticDataConfig& config)
{
  auto theta = compute_theta(true_params, base_params, desc);

  IdentificationData data;
  for (int t = 0; t <= config.duration_min; t += config.cgm_interval)
    data.cgm_time.push_back(static_cast<double>(t));
  data.cgm_value.assign(data.cgm_time.size(), 0.0);

  for (const auto& m : config.meals) {
    data.meal_time.push_back(m.first);
    data.meal_cho.push_back(m.second);
  }
  for (const auto& b : config.boluses) {
    data.bolus_time.push_back(b.first);
    data.bolus_dose.push_back(b.second);
  }
  for (const auto& i : config.insulin) {
    data.insulin_time.push_back(i.first);
    data.insulin_rate.push_back(i.second);
  }
  data.finalize();

  ObservationCost<Simulator> cost(data, base_params, desc);
  auto clean_cgm = cost.template simulate_observations<double>(theta.data());

  std::mt19937 rng(config.seed);
  std::normal_distribution<double> noise(0.0, config.noise_sd);
  for (size_t i = 0; i < clean_cgm.size(); ++i)
    data.cgm_value[i] = clean_cgm[i] + noise(rng);

  return data;
}

} // namespace uva_padova
