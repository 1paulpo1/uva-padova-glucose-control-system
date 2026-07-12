#include <gtest/gtest.h>
#include <cmath>

#include "uva_padova/cost_function.hpp"

using namespace uva_padova;

// ---------------------------------------------------------------------------
//  PriorCost: residuals
// ---------------------------------------------------------------------------
TEST(PriorCostTest, residuals) {
  std::vector<ParamDescriptor> desc = {
    {"p1", 0.0, 1.0},
    {"p2", 1.0, 2.0},
    {"p3", -1.0, 0.5},
  };
  PriorCost cost(desc);

  std::vector<double> theta = {0.5, 2.0, -2.0};
  std::vector<double> residuals(3);
  double* ptrs[] = {residuals.data()};
  const double* theta_ptr = theta.data();
  cost.Evaluate(&theta_ptr, residuals.data(), nullptr);

  EXPECT_NEAR(residuals[0], (0.5 - 0.0) / 1.0, 1e-12);
  EXPECT_NEAR(residuals[1], (2.0 - 1.0) / 2.0, 1e-12);
  EXPECT_NEAR(residuals[2], (-2.0 - (-1.0)) / 0.5, 1e-12);
}

// ---------------------------------------------------------------------------
//  PriorCost: Jacobian is diagonal 1/std_i
// ---------------------------------------------------------------------------
TEST(PriorCostTest, jacobian) {
  std::vector<ParamDescriptor> desc = {
    {"a", 0.0, 1.0},
    {"b", 0.0, 2.0},
    {"c", 0.0, 0.25},
  };
  PriorCost cost(desc);

  std::vector<double> theta = {0.1, 0.2, 0.3};
  std::vector<double> residuals(3);
  std::vector<double> J_buf(9);
  double* ptrs[] = {J_buf.data()};
  const double* theta_ptr = theta.data();
  cost.Evaluate(&theta_ptr, residuals.data(), ptrs);

  // Expect J[i,j] = δᵢⱼ / std_i
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double expected = (i == j) ? (1.0 / desc[i].prior_std) : 0.0;
      EXPECT_DOUBLE_EQ(J_buf[i * 3 + j], expected);
    }
  }
}

// ---------------------------------------------------------------------------
//  PriorCost: scalar prior
// ---------------------------------------------------------------------------
TEST(PriorCostTest, scalar_prior) {
  std::vector<ParamDescriptor> desc = {
    {"p1", 0.0, 1.0},
    {"p2", 1.0, 2.0},
  };
  PriorCost cost(desc);

  std::vector<double> theta = {0.5, 2.0};
  double r0 = (0.5 - 0.0) / 1.0;
  double r1 = (2.0 - 1.0) / 2.0;
  double expected = 0.5 * (r0 * r0 + r1 * r1);

  EXPECT_NEAR(cost.compute_scalar_prior(theta.data()), expected, 1e-12);
}

// ---------------------------------------------------------------------------
//  PriorCost: zero at prior mean
// ---------------------------------------------------------------------------
TEST(PriorCostTest, zero_at_mean) {
  std::vector<ParamDescriptor> desc = {
    {"p1", 2.0, 1.0},
    {"p2", -1.0, 2.0},
  };
  PriorCost cost(desc);

  std::vector<double> theta = {2.0, -1.0};
  std::vector<double> residuals(2);
  double* ptrs[] = {residuals.data()};
  const double* theta_ptr = theta.data();
  cost.Evaluate(&theta_ptr, residuals.data(), nullptr);

  EXPECT_NEAR(residuals[0], 0.0, 1e-12);
  EXPECT_NEAR(residuals[1], 0.0, 1e-12);
  EXPECT_NEAR(cost.compute_scalar_prior(theta.data()), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
//  PriorCost: n_residuals == n_params
// ---------------------------------------------------------------------------
TEST(PriorCostTest, num_residuals) {
  std::vector<ParamDescriptor> desc(5);
  PriorCost cost(desc);
  EXPECT_EQ(cost.num_residuals(), 5);
}
