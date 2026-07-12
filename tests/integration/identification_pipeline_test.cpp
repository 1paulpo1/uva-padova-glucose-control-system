#include <gtest/gtest.h>

#include "uva_padova/optimizer.hpp"
#include "test_data/synthetic_data_generator.hpp"

using namespace uva_padova;

// ===========================================================================
//  Synthetic data recovery: identify 3 params from simulated CGM
// ===========================================================================
TEST(IdentificationPipeline, synthetic_data_recovery) {
  Params true_p = Params::defaults_t1d();
  true_p.kp2  = 0.0030;
  true_p.Vm0  = 3.20;
  true_p.p2U  = 0.0280;

  SyntheticDataConfig cfg;
  cfg.duration_min = 720;
  cfg.noise_sd = 3.0;
  cfg.seed = 42;
  cfg.meals = {{60, 60000}, {360, 80000}, {600, 70000}};

  std::vector<ParamDescriptor> desc = {
    {"kp2", 0.0, 1.0, 0.0},
    {"Vm0", 0.0, 1.0, 0.0},
    {"p2U", 0.0, 1.0, 0.0},
  };

  IdentificationData data = generate_synthetic_data(
      true_p, Params::defaults_t1d(), desc, cfg);
  ASSERT_GT(data.cgm_time.size(), 0);

  auto result = IdentificationPipeline<>::run(
      data, Params::defaults_t1d(), desc, 3, 500);

  EXPECT_TRUE(result.success);

  Params recovered_p = result.to_params(Params::defaults_t1d(), desc);

  auto rel_err = [](double recovered, double true_val) {
    return std::abs(recovered - true_val) / true_val;
  };

  EXPECT_LT(rel_err(recovered_p.kp2, true_p.kp2), 0.50);
  EXPECT_LT(rel_err(recovered_p.Vm0, true_p.Vm0), 0.50);
  EXPECT_LT(rel_err(recovered_p.p2U, true_p.p2U), 0.50);
}
