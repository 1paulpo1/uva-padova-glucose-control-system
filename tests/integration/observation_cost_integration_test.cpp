#include <gtest/gtest.h>
#include <cmath>

#include "uva_padova/cost_function.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/integrals.hpp"
#include "test_data/synthetic_data.h"

using namespace uva_padova;

// ---------------------------------------------------------------------------
//  Integration test: real ODE, no NaN/Inf, doesn't crash
// ---------------------------------------------------------------------------
TEST(ObservationCostIntegrationTest, real_ode_sanity) {
  auto data = make_id_data({0.0, 30.0, 60.0},
                           {30.0}, {60000.0},
                           {}, {},
                           {0.0}, {100.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}, {"Vm0"}, {"p2U"}};
  auto base = Params::defaults_t1d();

  ObservationCost<> cost(data, base, descs);
  std::vector<double> theta = {0.0, 0.0, 0.0};

  // Evaluate without Jacobian
  std::vector<double> residuals(cost.n_obs_);
  const double* ptr = theta.data();
  EXPECT_TRUE(cost.Evaluate(&ptr, residuals.data(), nullptr));

  for (int i = 0; i < cost.n_obs_; ++i) {
    EXPECT_TRUE(std::isfinite(residuals[i]))
        << "non-finite residual at obs " << i;
  }

  // Evaluate with Jacobian
  std::vector<double> J_buf(cost.n_obs_ * cost.n_params_);
  double* J_ptrs[] = {J_buf.data()};
  EXPECT_TRUE(cost.Evaluate(&ptr, residuals.data(), J_ptrs));

  for (int i = 0; i < cost.n_obs_; ++i) {
    EXPECT_TRUE(std::isfinite(residuals[i]));
    for (int j = 0; j < cost.n_params_; ++j) {
      EXPECT_TRUE(std::isfinite(J_buf[i * cost.n_params_ + j]))
          << "non-finite Jacobian at obs=" << i << " param=" << j;
    }
  }

  // Scalar cost is finite
  double scalar = cost.compute_scalar_cost(theta.data());
  EXPECT_TRUE(std::isfinite(scalar));
  EXPECT_GE(scalar, 0.0);
}

// ---------------------------------------------------------------------------
//  Integration test: zero inputs, theta=0, Gsc near Gb
// ---------------------------------------------------------------------------
TEST(ObservationCostIntegrationTest, zero_input_near_gb) {
  // No meals, no boluses — just basal insulin
  auto data = make_id_data({0.0, 60.0, 120.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  auto base = Params::defaults_t1d();

  ObservationCost<> cost(data, base, descs);
  std::vector<double> theta = {0.0};

  auto cgm = cost.simulate_observations<double>(theta.data());

  ASSERT_EQ(cgm.size(), 3);
  // Basal glucose should be near Gb (120.0)
  // With default params and theta=0, Gb is 120 and steady state hits Gb
  EXPECT_NEAR(cgm[0], base.Gb, 5.0);
  // After 120 min with basal insulin, should stay near Gb
  EXPECT_NEAR(cgm[1], base.Gb, 10.0);
  EXPECT_NEAR(cgm[2], base.Gb, 10.0);
}

// ---------------------------------------------------------------------------
//  Integration test: compute_raw_gradients doesn't crash
// ---------------------------------------------------------------------------
TEST(ObservationCostIntegrationTest, raw_gradients_sanity) {
  auto data = make_id_data({0.0, 30.0, 60.0},
                           {30.0}, {60000.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}, {"Vm0"}, {"p2U"}};
  ObservationCost<> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta = {0.1, -0.05, 0.2};
  std::vector<double> raw_resid;
  Eigen::MatrixXd J_raw;
  EXPECT_TRUE(cost.compute_raw_gradients(theta.data(), raw_resid, J_raw));

  EXPECT_EQ(raw_resid.size(), cost.n_obs_);
  EXPECT_EQ(J_raw.rows(), cost.n_obs_);
  EXPECT_EQ(J_raw.cols(), cost.n_params_);

  for (double r : raw_resid)
    EXPECT_TRUE(std::isfinite(r));
  for (int i = 0; i < J_raw.size(); ++i)
    EXPECT_TRUE(std::isfinite(J_raw.data()[i]));
}

// ---------------------------------------------------------------------------
//  Gradient stability at extreme parameter values (large/small theta).
// ---------------------------------------------------------------------------
TEST(ObservationCostIntegrationTest, gradient_extreme_theta) {
  auto data = make_id_data({0.0, 60.0, 120.0},
                           {30.0}, {60000.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}, {"Vm0"}, {"p2U"}};
  ObservationCost<> cost(data, Params::defaults_t1d(), descs);

  // theta = 5.0 → exp(5.0) ≈ 148x multiplier
  // theta = -5.0 → exp(-5.0) ≈ 0.0067x multiplier
  std::vector<double> theta = {5.0, -5.0, 2.0};

  std::vector<double> residuals(cost.n_obs_);
  std::vector<double> J_buf(cost.n_obs_ * cost.n_params_);
  double* J_ptrs[] = {J_buf.data()};
  const double* tptr = theta.data();
  EXPECT_TRUE(cost.Evaluate(&tptr, residuals.data(), J_ptrs));

  for (int i = 0; i < cost.n_obs_; ++i)
    EXPECT_TRUE(std::isfinite(residuals[i])) << "residual " << i;
  for (int i = 0; i < cost.n_obs_ * cost.n_params_; ++i)
    EXPECT_TRUE(std::isfinite(J_buf[i])) << "J entry " << i;
}
