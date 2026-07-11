#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <cassert>

#include "uva_padova/types.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/integrals.hpp"
#include "uva_padova/delivery.hpp"
#include "uva_padova/cost_function.hpp"
#include "uva_padova/optimizer.hpp"

using namespace uva_padova;

// ---------------------------------------------------------------------------
// Generate synthetic data from a known parameter set
// ---------------------------------------------------------------------------
static IdentificationData generate_synthetic_data(const Params& true_params,
                                                  int duration_min,
                                                  double noise_sd,
                                                  int seed = 42,
                                                  bool with_bolus = false)
{
  // Find steady state — returns the matching basal insulin rate
  State ss;
  double insulin_ss = SteadyStateSolver::find(ss, true_params, true_params.Gb);

  // Create meal schedule
  struct Meal { int t; double cho_mg; };
  std::vector<Meal> meals = {
    { 60, 60000 },   // 60g breakfast at t=60
    { 360, 80000 },  // 80g lunch at t=360
    { 600, 70000 },  // 70g dinner at t=600
  };

  // Bolus schedule (add then subtract after duration)
  struct Bolus { int t; double rate; };
  std::vector<Bolus> boluses;
  if (with_bolus) {
    // Square-wave boluses: add at meal time, subtract 30 min later
    boluses = {
      {60,  0.8}, {90,  -0.8},   // breakfast
      {360, 1.0}, {390, -1.0},   // lunch
      {600, 0.9}, {630, -0.9},   // dinner
    };
  }

  // Simulate with agreement: basal = insulin_ss
  State x = ss;
  BasicInputs<double> u;
  u.insulin_rate = insulin_ss;

  std::vector<double> cgm_time, cgm_value;
  std::mt19937 rng(seed);
  std::normal_distribution<double> noise(0.0, noise_sd);

  std::vector<double> ins_time_vec, ins_rate_vec;
  int meal_idx = 0, bolus_idx = 0;
  for (int t = 0; t <= duration_min; ++t) {
    // Apply meal
    if (meal_idx < (int)meals.size() && meals[meal_idx].t == t) {
      x.Qsto1 += meals[meal_idx].cho_mg;
      x.mealD  = meals[meal_idx].cho_mg;
      meal_idx++;
    }

    // Apply bolus events
    while (bolus_idx < (int)boluses.size() && boluses[bolus_idx].t == t) {
      u.insulin_rate += boluses[bolus_idx].rate;
      bolus_idx++;
    }

    // Record CGM every 5 min
    if (t % 5 == 0) {
      cgm_time.push_back((double)t);
      cgm_value.push_back(x.Gsc + noise(rng));
    }

    // Step
    if (t < duration_min) {
      RK4Integrator::step(UvaPadovaODESystem{}, x, u, true_params, 1.0);
    }
  }

  IdentificationData data;
  data.cgm_time      = cgm_time;
  data.cgm_value     = cgm_value;
  data.meal_time.clear();
  data.meal_cho.clear();
  for (auto& m : meals) {
    data.meal_time.push_back((double)m.t);
    data.meal_cho.push_back(m.cho_mg);
  }
  // Pack insulin events (boluses) for the identification pipeline
  data.insulin_time.clear();
  data.insulin_rate.clear();
  if (with_bolus) {
    for (auto& b : boluses) {
      data.insulin_time.push_back((double)b.t);
      data.insulin_rate.push_back(b.rate);
    }
  }
  data.finalize();

  return data;
}

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------
int main() {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "=== Identification Pipeline Test ===\n\n";

  // --- Ground truth parameters (T1D) ---
  Params true_p = Params::defaults_t1d();
  // Modify a few params to create a distinct subject
  true_p.kp2  = 0.0030;    // ~1.43x default
  true_p.Vm0  = 3.20;      // ~1.28x default
  true_p.p2U  = 0.0280;    // ~0.85x default

  std::cout << "Ground truth params:\n";
  std::cout << "  kp2 = " << true_p.kp2 << "\n";
  std::cout << "  Vm0 = " << true_p.Vm0 << "\n";
  std::cout << "  p2U = " << true_p.p2U << "\n\n";

  // --- Generate synthetic data ---
  const int duration_min = 720;  // 12 hours, 3 meals
  const double noise_sd = 3.0;   // mg/dL

  std::cout << "Generating synthetic data (duration=" << duration_min
            << "min, noise_sd=" << noise_sd << "mg/dL)...\n";
  IdentificationData data = generate_synthetic_data(true_p, duration_min, noise_sd);
  std::cout << "  Observations: " << data.cgm_time.size() << "\n";
  std::cout << "  Total min: " << data.total_min() << "\n\n";

  // --- Base params (keep default, identification will adjust) ---
  Params base_p = Params::defaults_t1d();

  // --- Parameter descriptors (which params to identify) ---
  std::vector<ParamDescriptor> desc = {
    { "kp2", /*prior*/ 0.0, /*std*/ 1.0 /*log*/, 0.0 },
    { "Vm0", 0.0, 1.0, 0.0 },
    { "p2U", 0.0, 1.0, 0.0 },
  };

  std::cout << "Starting identification (3 CMA-ES restarts + LM)...\n\n";

  // --- Run pipeline ---
  auto result = IdentificationPipeline::run(data, base_p, desc, 3, 500);

  // --- Report ---
  std::cout << "\n=== Results ===\n";
  std::cout << "  Success: " << (result.success ? "yes" : "no") << "\n";
  std::cout << "  Final cost: " << result.cost << "\n";
  std::cout << "  Total evals: " << result.total_evals << "\n";

  Params recovered_p = result.to_params(base_p, desc);

  double rel_err_kp2 = std::abs(recovered_p.kp2 - true_p.kp2) / true_p.kp2;
  double rel_err_Vm0 = std::abs(recovered_p.Vm0 - true_p.Vm0) / true_p.Vm0;
  double rel_err_p2U = std::abs(recovered_p.p2U - true_p.p2U) / true_p.p2U;

  std::cout << "\n  Recovered kp2: " << recovered_p.kp2
            << " (true=" << true_p.kp2 << ", rel_err=" << rel_err_kp2 << ")\n";
  std::cout << "  Recovered Vm0: " << recovered_p.Vm0
            << " (true=" << true_p.Vm0 << ", rel_err=" << rel_err_Vm0 << ")\n";
  std::cout << "  Recovered p2U: " << recovered_p.p2U
            << " (true=" << true_p.p2U << ", rel_err=" << rel_err_p2U << ")\n";

  // --- Check: all params within 50% relative error ---
  bool all_ok = true;
  if (rel_err_kp2 > 0.50) { std::cerr << "FAIL: kp2 error too large\n"; all_ok = false; }
  if (rel_err_Vm0 > 0.50) { std::cerr << "FAIL: Vm0 error too large\n"; all_ok = false; }
  if (rel_err_p2U > 0.50) { std::cerr << "FAIL: p2U error too large\n"; all_ok = false; }

  if (all_ok) {
    std::cout << "\n*** ALL CHECKS PASSED ***\n";
    return 0;
  } else {
    std::cerr << "\n*** SOME CHECKS FAILED ***\n";
    return 1;
  }
}
