#include <gtest/gtest.h>
#include <functional>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Core>

#include "uva_padova/optimizer.hpp"
#include "test_data/synthetic_data.h"
#include "mocks/mock_simulator.h"

using namespace uva_padova;

// ===========================================================================
// Mock optimizers for testing Pipeline logic
// ===========================================================================
struct MockCMAES {
  struct Result {
    std::vector<double> best_params;
    double best_cost{1e20};
    int n_evals{0};
  };

  static inline std::vector<double> costs;
  static inline std::vector<double> params_flat;
  static inline int param_dim{1};
  static inline int n_evals_per_call{100};
  static inline int call_count{0};

  static Result run(std::function<double(const double*, int)> /*fit_func*/,
                    const std::vector<double>& /*initial_sigma*/,
                    int /*max_evals*/,
                    int /*seed*/,
                    double /*lower_bound*/,
                    double /*upper_bound*/) {
    int i = call_count++;
    std::vector<double> p(param_dim, 0.0);
    if (!params_flat.empty()) {
      for (int j = 0; j < param_dim && j < static_cast<int>(params_flat.size()); ++j)
        p[j] = params_flat[j];
    }
    double c = (i < static_cast<int>(costs.size())) ? costs[i] : 1e20;
    return {p, c, n_evals_per_call};
  }

  static void reset() {
    costs.clear();
    params_flat.clear();
    param_dim = 1;
    n_evals_per_call = 100;
    call_count = 0;
  }
};

struct MockLM {
  struct Result {
    std::vector<double> final_params;
    double final_cost{1e20};
    int n_iters{0};
    bool converged{false};
    Eigen::MatrixXd cov;
    Eigen::MatrixXd cov_sandwich;
  };

  struct CovarianceResult {
    Eigen::MatrixXd cov;
    Eigen::MatrixXd cov_sandwich;
  };

  static inline std::vector<double> next_params;
  static inline double next_cost{0.0};
  static inline int next_n_iters{5};
  static inline bool next_converged{true};
  static inline bool use_self_cov{true};
  static inline int call_count{0};

  static Result run(const std::vector<double>& initial_params,
                    ObservationCost<>& /*obs_cost*/,
                    PriorCost& /*prior_cost*/,
                    int /*max_iters*/) {
    ++call_count;
    int n = static_cast<int>(initial_params.size());
    auto params = next_params.empty() ? initial_params : next_params;
    auto cov = use_self_cov ? Eigen::MatrixXd::Identity(n, n) : Eigen::MatrixXd{};
    return {params, next_cost, next_n_iters, next_converged, cov, {}};
  }

  static CovarianceResult compute_covariance(
      const std::vector<double>& theta,
      ObservationCost<>& /*obs_cost*/,
      PriorCost& /*prior_cost*/) {
    int n = static_cast<int>(theta.size());
    return {Eigen::MatrixXd::Identity(n, n), {}};
  }

  static void reset() {
    next_params.clear();
    next_cost = 0.0;
    next_n_iters = 5;
    next_converged = true;
    use_self_cov = true;
    call_count = 0;
  }
};

// ===========================================================================
// Pipeline<Mock,Mock> — restart selection logic
// ===========================================================================
TEST(PipelineMockTest, selects_best_restart) {
  MockCMAES::reset();
  MockLM::reset();
  MockCMAES::param_dim = 2;
  MockCMAES::costs = {10.0, 5.0, 8.0};
  MockLM::next_cost = 20.0;  // LM worse than CMA-ES best, so Pipeline keeps CMA best

  auto data = make_id_data({10.0, 20.0});
  std::vector<ParamDescriptor> desc = {{"kp1", 0, 1}, {"kp2", 0, 1}};

  auto result = IdentificationPipeline<MockCMAES, MockLM>::run(
      data, Params::defaults_t1d(), desc, 3, 100);

  EXPECT_TRUE(result.success);
  // Best CMA-ES cost = min{10, 5, 8} = 5; LM cost = 20, so result keeps CMA-ES cost
  EXPECT_DOUBLE_EQ(result.cost, 5.0);
  ASSERT_EQ(result.all_restarts.size(), 3);
  EXPECT_DOUBLE_EQ(result.all_restarts[0].cost, 10.0);
  EXPECT_DOUBLE_EQ(result.all_restarts[1].cost, 5.0);
  EXPECT_DOUBLE_EQ(result.all_restarts[2].cost, 8.0);
}

TEST(PipelineMockTest, all_restarts_fail) {
  MockCMAES::reset();
  MockLM::reset();
  MockCMAES::param_dim = 1;
  MockCMAES::costs = {1e20, 1e20, 1e20};
  MockLM::next_cost = 1e20;

  auto data = make_id_data({10.0});
  std::vector<ParamDescriptor> desc = {{"kp1", 0, 1}};

  auto result = IdentificationPipeline<MockCMAES, MockLM>::run(
      data, Params::defaults_t1d(), desc, 3, 100);

  EXPECT_FALSE(result.success);
  EXPECT_GE(result.cost, 1e19);
}

TEST(PipelineMockTest, lm_improves_over_cmaes) {
  MockCMAES::reset();
  MockLM::reset();
  MockCMAES::param_dim = 1;
  MockCMAES::costs = {10.0, 12.0, 8.0};  // best CMA = 8.0
  MockLM::next_cost = 0.1;                // LM improves

  auto data = make_id_data({10.0});
  std::vector<ParamDescriptor> desc = {{"kp1", 0, 1}};

  auto result = IdentificationPipeline<MockCMAES, MockLM>::run(
      data, Params::defaults_t1d(), desc, 3, 100);

  EXPECT_TRUE(result.success);
  EXPECT_DOUBLE_EQ(result.cost, 0.1);
}

TEST(PipelineMockTest, to_params_applies_exponential_multipliers) {
  MockCMAES::reset();
  MockLM::reset();
  MockCMAES::param_dim = 2;
  MockCMAES::params_flat = {0.5, -0.2};
  MockCMAES::costs = {5.0};        // ensure best_cma_cost < 1e19 so best_cma_point updates

  auto data = make_id_data({10.0, 20.0});
  std::vector<ParamDescriptor> desc = {{"kp1", 0, 1}, {"kp2", 0, 1}};

  auto result = IdentificationPipeline<MockCMAES, MockLM>::run(
      data, Params::defaults_t1d(), desc, 1, 100);

  Params base = Params::defaults_t1d();
  Params recovered = result.to_params(base, desc);

  EXPECT_NEAR(recovered.kp1, base.kp1 * std::exp(0.5), 1e-12);
  EXPECT_NEAR(recovered.kp2, base.kp2 * std::exp(-0.2), 1e-12);
}

TEST(PipelineMockTest, lm_not_used_when_not_converged) {
  MockCMAES::reset();
  MockLM::reset();
  MockCMAES::param_dim = 1;
  MockCMAES::costs = {10.0, 8.0};
  MockLM::next_cost = 5.0;          // LM has lower cost
  MockLM::next_converged = false;   // but did not converge
  // With cmaes_failed=false and !lm_res.converged, condition is false → keep CMA best

  auto data = make_id_data({10.0});
  std::vector<ParamDescriptor> desc = {{"kp1", 0, 1}};

  auto result = IdentificationPipeline<MockCMAES, MockLM>::run(
      data, Params::defaults_t1d(), desc, 2, 100);

  EXPECT_TRUE(result.success);
  EXPECT_DOUBLE_EQ(result.cost, 8.0);  // CMA best, not LM
}

// ===========================================================================
// CMAESOptimizer — integration-light with LinearMockSimulator
// ===========================================================================
class CMAESOptimizerTest : public ::testing::Test {
protected:
  void SetUp() override {
    LinearMockSimulator::reset();
  }

  auto make_cost(const std::vector<ParamDescriptor>& desc = {{"kp1", 0, 1}}) {
    auto data = make_id_data({10.0, 20.0, 30.0, 40.0, 50.0});
    return ObservationCost<LinearMockSimulator>(data, Params::defaults_t1d(), desc);
  }

  auto make_prior(const std::vector<ParamDescriptor>& desc = {{"kp1", 0, 1}}) {
    return PriorCost(desc);
  }
};

TEST_F(CMAESOptimizerTest, converges_on_quadratic) {
  auto obs_cost = make_cost();
  auto prior = make_prior();

  std::function<double(const double*, int)> fit =
      [&](const double* x, int) -> double {
    double f = obs_cost.compute_scalar_cost(x) + prior.compute_scalar_prior(x);
    return std::isfinite(f) ? f : 1e20;
  };

  auto res = CMAESOptimizer::run(fit, {1.0}, 500, 42);

  EXPECT_TRUE(std::isfinite(res.best_cost));
  EXPECT_EQ(res.best_params.size(), 1);
}

TEST_F(CMAESOptimizerTest, seed_reproducibility) {
  auto obs_cost = make_cost();
  auto prior = make_prior();

  std::function<double(const double*, int)> fit =
      [&](const double* x, int) -> double {
    double f = obs_cost.compute_scalar_cost(x) + prior.compute_scalar_prior(x);
    return std::isfinite(f) ? f : 1e20;
  };

  auto r1 = CMAESOptimizer::run(fit, {1.0}, 300, 42);
  auto r2 = CMAESOptimizer::run(fit, {1.0}, 300, 42);

  ASSERT_EQ(r1.best_params.size(), r2.best_params.size());
  ASSERT_FALSE(r1.best_params.empty());
  EXPECT_NEAR(r1.best_params[0], r2.best_params[0], 1e-10);
  EXPECT_NEAR(r1.best_cost, r2.best_cost, 1e-10);
}

TEST_F(CMAESOptimizerTest, max_evals_respected) {
  auto obs_cost = make_cost();
  auto prior = make_prior();

  std::function<double(const double*, int)> fit =
      [&](const double* x, int) -> double {
    double f = obs_cost.compute_scalar_cost(x) + prior.compute_scalar_prior(x);
    return std::isfinite(f) ? f : 1e20;
  };

  const int max_evals = 100;
  auto res = CMAESOptimizer::run(fit, {1.0}, max_evals, 42);
  EXPECT_LE(res.n_evals, max_evals);
}

TEST_F(CMAESOptimizerTest, bounds_respected_when_minimum_outside) {
  std::function<double(const double*, int)> fit =
      [](const double* x, int) -> double {
    return (x[0]-2)*(x[0]-2) + (x[1]+2)*(x[1]+2);
  };

  auto res = CMAESOptimizer::run(fit, {1.0, 1.0}, 2000, 42, -1.0, 1.0);

  EXPECT_TRUE(std::isfinite(res.best_cost));
  ASSERT_EQ(res.best_params.size(), 2);
  EXPECT_GE(res.best_params[0], -1.0);
  EXPECT_LE(res.best_params[0],  1.0);
  EXPECT_GE(res.best_params[1], -1.0);
  EXPECT_LE(res.best_params[1],  1.0);
  EXPECT_LT(res.best_cost, 50.0);
}

TEST_F(CMAESOptimizerTest, different_param_scales) {
  std::function<double(const double*, int)> fit =
      [](const double* x, int) -> double {
    double a = 100.0 * x[0];
    double b = 0.01 * x[1];
    return a*a + b*b;
  };

  auto res = CMAESOptimizer::run(fit, {1.0, 1.0}, 3000, 42);

  EXPECT_TRUE(std::isfinite(res.best_cost));
  ASSERT_EQ(res.best_params.size(), 2);
  EXPECT_LT(res.best_cost, 100.0);
}

TEST_F(CMAESOptimizerTest, twenty_params_quadratic) {
  const int dim = 20;
  std::mt19937 rng(123);
  std::normal_distribution<> nd(0.0, 1.0);

  Eigen::MatrixXd A(dim, dim);
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      A(i, j) = nd(rng);
  Eigen::MatrixXd H = A.transpose() * A
                    + 0.1 * Eigen::MatrixXd::Identity(dim, dim);

  Eigen::VectorXd true_x(dim);
  for (int i = 0; i < dim; ++i) true_x(i) = nd(rng) * 0.5;

  std::function<double(const double*, int)> fit =
      [&H, &true_x](const double* x, int n) -> double {
    Eigen::Map<const Eigen::VectorXd> xm(x, n);
    Eigen::VectorXd d = xm - true_x;
    return d.transpose() * H * d;
  };

  auto res = CMAESOptimizer::run(fit, std::vector<double>(dim, 1.0), 10000, 42);

  EXPECT_TRUE(std::isfinite(res.best_cost));
  ASSERT_EQ(res.best_params.size(), dim);

  Eigen::Map<const Eigen::VectorXd> best(res.best_params.data(), dim);
  double true_cost = fit(true_x.data(), dim);
  EXPECT_EQ(true_cost, 0.0);
  EXPECT_GT(res.best_cost, 0.0);

  // Must beat a random guess drawn from ~[-3, 3]
  std::vector<double> rnd(dim);
  for (int i = 0; i < dim; ++i) rnd[i] = nd(rng) * 3.0;
  double rnd_cost = fit(rnd.data(), dim);
  EXPECT_LT(res.best_cost, rnd_cost);
}

TEST_F(CMAESOptimizerTest, different_initial_sigma_similar_result) {
  std::function<double(const double*, int)> fit =
      [](const double* x, int) -> double {
    return x[0]*x[0] + x[1]*x[1] + 5.0;
  };

  auto r1 = CMAESOptimizer::run(fit, {1.0, 1.0}, 1000, 42);
  auto r2 = CMAESOptimizer::run(fit, {0.1, 3.0}, 1000, 42);

  EXPECT_TRUE(std::isfinite(r1.best_cost));
  EXPECT_TRUE(std::isfinite(r2.best_cost));
  EXPECT_NEAR(r1.best_cost, r2.best_cost, r1.best_cost * 0.5 + 1.0);
}

// ===========================================================================
// LMOptimizer — integration-light with LinearMockSimulator
// ===========================================================================
class LMOptimizerTest : public ::testing::Test {
protected:
  void SetUp() override {
    LinearMockSimulator::reset();
  }
};

TEST_F(LMOptimizerTest, converges_with_prior) {
  auto data = make_id_data({10.0, 20.0, 30.0, 40.0, 50.0});
  ObservationCost<LinearMockSimulator> obs_cost(data, Params::defaults_t1d(), {{"kp1", 0, 1}});
  PriorCost prior({{"kp1", 0, 1}});

  auto res = LMOptimizer::run({0.0}, obs_cost, prior, 50);

  EXPECT_TRUE(res.converged);
  EXPECT_TRUE(std::isfinite(res.final_cost));
  EXPECT_EQ(res.final_params.size(), 1);
}

TEST_F(LMOptimizerTest, max_iters_limits) {
  auto data = make_id_data({10.0, 20.0, 30.0});
  ObservationCost<LinearMockSimulator> obs_cost(data, Params::defaults_t1d(), {{"kp1", 0, 1}});
  PriorCost prior({{"kp1", 0, 1}});

  auto res = LMOptimizer::run({1.0}, obs_cost, prior, 1);

  EXPECT_FALSE(res.converged);
  // Ceres includes the initial state in iterations, so 1 max_iter → n_iters ≥ 1
  EXPECT_GE(res.n_iters, 1);
}

TEST_F(LMOptimizerTest, covariance_positive_diagonal) {
  auto data = make_id_data({10.0, 20.0, 30.0, 40.0, 50.0, 60.0});
  ObservationCost<LinearMockSimulator> obs_cost(data, Params::defaults_t1d(), {{"kp1", 0, 1}});
  PriorCost prior({{"kp1", 0, 1}});

  auto res = LMOptimizer::run({0.0}, obs_cost, prior, 50);
  ASSERT_TRUE(res.converged);
  ASSERT_GT(res.cov.rows(), 0);
  EXPECT_GT(res.cov(0, 0), 0.0);

  // Also test standalone compute_covariance at the same point
  auto cov_res = LMOptimizer::compute_covariance(res.final_params, obs_cost, prior);
  ASSERT_GT(cov_res.cov.rows(), 0);
  EXPECT_GT(cov_res.cov(0, 0), 0.0);
}
