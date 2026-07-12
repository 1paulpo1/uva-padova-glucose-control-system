#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "uva_padova/types.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/integrals.hpp"
#include "uva_padova/cost_function.hpp"
#include "uva_padova/optimizer.hpp"

using namespace uva_padova;

namespace {

IdentificationData generate_synthetic_data(const Params& true_params,
                                           int duration_min,
                                           double noise_sd,
                                           int seed = 42) {
  State ss;
  double insulin_ss = SteadyStateSolver::find(ss, true_params, true_params.Gb);

  struct Meal { int t; double cho_mg; };
  std::vector<Meal> meals = {
    { 60, 60000 },
    { 360, 80000 },
    { 600, 70000 },
  };

  State x = ss;
  BasicInputs<double> u;
  u.insulin_rate = insulin_ss;

  std::vector<double> cgm_time, cgm_value;
  std::mt19937 rng(seed);
  std::normal_distribution<double> noise(0.0, noise_sd);

  int meal_idx = 0;
  for (int t = 0; t <= duration_min; ++t) {
    if (meal_idx < (int)meals.size() && meals[meal_idx].t == t) {
      x.Qsto1 += meals[meal_idx].cho_mg;
      x.mealD  = meals[meal_idx].cho_mg;
      meal_idx++;
    }

    if (t % 5 == 0) {
      cgm_time.push_back((double)t);
      cgm_value.push_back(x.Gsc + noise(rng));
    }

    if (t < duration_min) {
      RK4Integrator::step(UvaPadovaODESystem{}, x, u, true_params, 1.0);
    }
  }

  IdentificationData data;
  data.cgm_time  = cgm_time;
  data.cgm_value = cgm_value;
  for (auto& m : meals) {
    data.meal_time.push_back((double)m.t);
    data.meal_cho.push_back(m.cho_mg);
  }
  data.finalize();
  return data;
}

} // anonymous namespace

// ===========================================================================
//  Synthetic data recovery: identify 3 params from simulated CGM
// ===========================================================================
TEST(IdentificationPipeline, synthetic_data_recovery) {
  Params true_p = Params::defaults_t1d();
  true_p.kp2  = 0.0030;
  true_p.Vm0  = 3.20;
  true_p.p2U  = 0.0280;

  const int duration_min = 720;
  const double noise_sd = 3.0;

  IdentificationData data = generate_synthetic_data(true_p, duration_min, noise_sd);
  ASSERT_GT(data.cgm_time.size(), 0);

  Params base_p = Params::defaults_t1d();

  std::vector<ParamDescriptor> desc = {
    { "kp2", 0.0, 1.0, 0.0 },
    { "Vm0", 0.0, 1.0, 0.0 },
    { "p2U", 0.0, 1.0, 0.0 },
  };

  auto result = IdentificationPipeline<>::run(data, base_p, desc, 3, 500);

  EXPECT_TRUE(result.success);

  Params recovered_p = result.to_params(base_p, desc);
  auto rel_err = [](double recovered, double true_val) {
    return std::abs(recovered - true_val) / true_val;
  };

  EXPECT_LT(rel_err(recovered_p.kp2, true_p.kp2), 0.50);
  EXPECT_LT(rel_err(recovered_p.Vm0, true_p.Vm0), 0.50);
  EXPECT_LT(rel_err(recovered_p.p2U, true_p.p2U), 0.50);
}
