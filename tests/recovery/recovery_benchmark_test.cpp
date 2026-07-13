#include <gtest/gtest.h>
#include <vector>
#include <string>

#include "uva_padova/optimizer.hpp"
#include "uva_padova/synthetic_data_generator.hpp"

using namespace uva_padova;

// ---------------------------------------------------------------------------
// Build a meal schedule for multi-day simulations (3 meals/day)
// ---------------------------------------------------------------------------
static std::vector<std::pair<double, double>> daily_meals(int duration_min) {
  std::vector<std::pair<double, double>> meals;
  int n_days = duration_min / 1440 + 1;
  for (int day = 0; day < n_days; ++day) {
    int offset = day * 1440;
    meals.emplace_back(offset + 360, 60000);
    meals.emplace_back(offset + 720, 80000);
    meals.emplace_back(offset + 1080, 70000);
  }
  return meals;
}

// ---------------------------------------------------------------------------
// Run a single recovery scenario
// ---------------------------------------------------------------------------
static void run_recovery(int duration_min, int cgm_interval,
                         double noise_sd,
                         int cmaes_evals, int n_restarts,
                         const std::vector<std::pair<std::string, double>>& true_params,
                         double max_rel_error,
                         const std::string& label) {
  Params true_p = Params::defaults_t1d();
  for (const auto& [name, val] : true_params) {
    if (name == "kp2")  true_p.kp2  = val;
    if (name == "Vm0")  true_p.Vm0  = val;
    if (name == "p2U")  true_p.p2U  = val;
    if (name == "ka1")  true_p.ka1  = val;
    if (name == "ka2")  true_p.ka2  = val;
    if (name == "kd")   true_p.kd   = val;
    if (name == "kabs") true_p.kabs = val;
    if (name == "VG")   true_p.VG   = val;
  }

  SyntheticDataConfig gen_cfg;
  gen_cfg.duration_min = duration_min;
  gen_cfg.cgm_interval = cgm_interval;
  gen_cfg.noise_sd = noise_sd;
  gen_cfg.seed = 42;
  gen_cfg.meals = daily_meals(duration_min);

  std::vector<ParamDescriptor> desc;
  for (const auto& [name, val] : true_params) {
    (void)val;
    desc.push_back({name, 0.0, 1.0, 0.0});
  }

  auto gen_result = generate_synthetic_data(
      true_p, Params::defaults_t1d(), desc, gen_cfg);
  IdentificationData& data = gen_result.data;
  ASSERT_GT(data.cgm_time.size(), 0);

  auto result = IdentificationPipeline<>::run(
      data, Params::defaults_t1d(), desc, n_restarts, cmaes_evals);

  Params recovered_p = result.to_params(Params::defaults_t1d(), desc);

  auto rel_err = [](double recovered, double true_val) {
    return std::abs(recovered - true_val) / true_val;
  };

  bool any_failed = false;
  for (const auto& [name, val] : true_params) {
    double recovered_val = param_value(recovered_p, name);
    double err = rel_err(recovered_val, val);
    std::cout << "  " << label << " " << name
              << ": true=" << val
              << " recovered=" << recovered_val
              << " rel_err=" << err
              << " (threshold=" << max_rel_error << ")\n";
    if (err > max_rel_error) any_failed = true;
  }

  EXPECT_FALSE(any_failed)
      << label << ": parameter recovery exceeded threshold "
      << max_rel_error << "\n"
      << "pipeline success=" << (result.success ? "true" : "false")
      << " cost=" << result.cost;
}

// ===========================================================================
// Benchmark: recovery accuracy improves with more data / less noise
// ===========================================================================
//
// Each test perturbs 3 default parameters (kp2, Vm0, p2U) to known true
// values, generates synthetic CGM, runs the full identification pipeline,
// and checks that relative recovery errors stay below a threshold.
//
// Thresholds are set empirically — tighten as the algorithm improves.
// ===========================================================================

// 12h, 5-min CGM, 500 evals, 3 restarts — matches original integration test
TEST(RecoveryBenchmark, baseline) {
  run_recovery(720, 5, 3.0, 500, 3,
               {{"kp2", 0.0030}, {"Vm0", 3.20}, {"p2U", 0.0280}},
               0.50, "baseline");
}

// 1 day, 15-min CGM, 1000 evals, 5 restarts, moderate noise
TEST(RecoveryBenchmark, oneday_moderate_noise) {
  run_recovery(1440, 15, 3.0, 1000, 5,
               {{"kp2", 0.0030}, {"Vm0", 3.20}, {"p2U", 0.0280}},
               0.35, "1d_moderate_noise");
}

// 1 day, 15-min CGM, 1000 evals, 5 restarts, zero noise
TEST(RecoveryBenchmark, oneday_zero_noise) {
  run_recovery(1440, 15, 0.0, 1000, 5,
               {{"kp2", 0.0030}, {"Vm0", 3.20}, {"p2U", 0.0280}},
               0.25, "1d_zero_noise");
}

// 2 days, 15-min CGM, 1000 evals, 5 restarts, zero noise
TEST(RecoveryBenchmark, twoday_zero_noise) {
  run_recovery(2880, 15, 0.0, 1000, 5,
               {{"kp2", 0.0030}, {"Vm0", 3.20}, {"p2U", 0.0280}},
               0.20, "2d_zero_noise");
}


