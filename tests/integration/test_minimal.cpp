#include <gtest/gtest.h>
#include <cmath>
#include "uva_padova/cost_function.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/integrals.hpp"
#include "test_data/synthetic_data.h"
using namespace uva_padova;
TEST(ObservationCostIntegrationTest, gradient_convergence_minimal) {
  auto data = make_id_data({0.0, 60.0, 120.0}, {30.0}, {60000.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}, {"Vm0"}, {"p2U"}};
  ObservationCost<> cost(data, Params::defaults_t1d(), descs);
  std::vector<double> theta = {0.1, -0.05, 0.02};
  std::vector<double> r(cost.n_obs_), J(cost.n_obs_ * cost.n_params_);
  double* Jp[] = {J.data()};
  const double* tp = theta.data();
  EXPECT_TRUE(cost.Evaluate(&tp, r.data(), Jp));
  for (auto x : r) EXPECT_TRUE(std::isfinite(x));
  for (auto x : J) EXPECT_TRUE(std::isfinite(x));
}
