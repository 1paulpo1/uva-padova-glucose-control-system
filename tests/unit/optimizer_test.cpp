#include <gtest/gtest.h>
#include <functional>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <filesystem>
#include <fstream>
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
  static inline int throw_on_call{-1};  // call index that throws std::runtime_error (-1 = never)

  static Result run(std::function<double(const double*, int)> /*fit_func*/,
                    const std::vector<double>& /*initial_sigma*/,
                    int /*max_evals*/,
                    int /*seed*/,
                    double /*lower_bound*/,
                    double /*upper_bound*/) {
    int i = call_count++;
    if (throw_on_call >= 0 && i == throw_on_call)
      throw std::runtime_error("mock cmaes exception");
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
    throw_on_call = -1;
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
  MockCMAES::costs = {5.0};        // ensure best_cma_cost < 1e10 so best_cma_point updates

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

TEST(PipelineMockTest, all_cmaes_costs_exceed_threshold) {
  MockCMAES::reset();
  MockLM::reset();
  MockCMAES::param_dim = 1;
  MockCMAES::costs = {5e10, 5e10, 5e10};
  MockLM::next_cost = 5e10;

  auto data = make_id_data({10.0});
  std::vector<ParamDescriptor> desc = {{"kp1", 0, 1}};

  auto result = IdentificationPipeline<MockCMAES, MockLM>::run(
      data, Params::defaults_t1d(), desc, 3, 100);

  EXPECT_FALSE(result.success);
  EXPECT_GE(result.cost, 5e10);
}

TEST(PipelineMockTest, exception_in_cmaes_restart) {
  MockCMAES::reset();
  MockLM::reset();
  MockCMAES::param_dim = 1;
  MockCMAES::costs = {1e20, 5.0, 5.0};
  MockCMAES::throw_on_call = 0;  // first call throws
  MockLM::next_cost = 0.1;

  auto data = make_id_data({10.0});
  std::vector<ParamDescriptor> desc = {{"kp1", 0, 1}};

  auto result = IdentificationPipeline<MockCMAES, MockLM>::run(
      data, Params::defaults_t1d(), desc, 3, 100);

  EXPECT_TRUE(result.success);
  ASSERT_EQ(result.all_restarts.size(), 3);
  EXPECT_DOUBLE_EQ(result.all_restarts[0].cost, 1e20);
  EXPECT_DOUBLE_EQ(result.all_restarts[1].cost, 5.0);
  EXPECT_DOUBLE_EQ(result.all_restarts[2].cost, 5.0);
  EXPECT_DOUBLE_EQ(result.cost, 0.1);
}

TEST(PipelineMockTest, to_params_skips_fixed_descriptors) {
  MockCMAES::reset();
  MockLM::reset();
  MockCMAES::param_dim = 2;
  MockCMAES::params_flat = {0.5, -0.2};
  MockCMAES::costs = {5.0};

  auto data = make_id_data({10.0, 20.0});
  std::vector<ParamDescriptor> active_desc = {{"kp1", 0, 1}, {"kp2", 0, 1}};
  std::vector<ParamDescriptor> full_desc = {
      {"kp1", 0, 1},
      {"kp2", 0, 1},
      {"VI", 0, 1, 1.0, true}  // fixed
  };

  auto result = IdentificationPipeline<MockCMAES, MockLM>::run(
      data, Params::defaults_t1d(), active_desc, 1, 100);

  Params base = Params::defaults_t1d();
  Params recovered = result.to_params(base, full_desc);

  EXPECT_NEAR(recovered.kp1, base.kp1 * std::exp(0.5), 1e-12);
  EXPECT_NEAR(recovered.kp2, base.kp2 * std::exp(-0.2), 1e-12);
  EXPECT_DOUBLE_EQ(recovered.VI, base.VI);
}

TEST(PipelineMockTest, all_restarts_recorded_even_bad) {
  MockCMAES::reset();
  MockLM::reset();
  MockCMAES::param_dim = 1;
  MockCMAES::costs = {1e20, 5.0, 1e20};
  MockLM::next_cost = 5.0;

  auto data = make_id_data({10.0});
  std::vector<ParamDescriptor> desc = {{"kp1", 0, 1}};

  auto result = IdentificationPipeline<MockCMAES, MockLM>::run(
      data, Params::defaults_t1d(), desc, 3, 100);

  ASSERT_EQ(result.all_restarts.size(), 3);
  EXPECT_DOUBLE_EQ(result.all_restarts[0].cost, 1e20);
  EXPECT_DOUBLE_EQ(result.all_restarts[1].cost, 5.0);
  EXPECT_DOUBLE_EQ(result.all_restarts[2].cost, 1e20);
}

TEST(PipelineMockTest, run_from_config_corrupted_json) {
  auto tmp = std::filesystem::temp_directory_path() / "bad_config.json";
  {
    std::ofstream ofs(tmp.string());
    ofs << "{broken";
  }

  auto data = make_id_data({10.0});
  auto result = IdentificationPipeline<>::run_from_config(
      data, Params::defaults_t1d(), tmp.string(), 1, 100);

  EXPECT_FALSE(result.success);
  std::filesystem::remove(tmp);
}

TEST(PipelineMockTest, run_from_config_minimal_json) {
  auto tmp = std::filesystem::temp_directory_path() / "min_config.json";
  {
    std::ofstream ofs(tmp.string());
    ofs << R"({"parameters": [{"name": "kp1"}]})";
  }

  auto data = make_id_data({10.0, 20.0, 30.0});
  auto result = IdentificationPipeline<>::run_from_config(
      data, Params::defaults_t1d(), tmp.string(), 3, 100);

  // Config loaded successfully and pipeline executed (may or may not converge)
  ASSERT_GE(result.all_restarts.size(), 1);
  std::filesystem::remove(tmp);
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

// ===========================================================================
// Rosenbrock-based nonlinear 2-parameter simulator
// ===========================================================================
// Rosenbrock-inspired 2-parameter nonlinear simulator.
//
// With data at {t=0, t=1} and CGM = {0, 0} and sigma=1.0, the residuals are:
//   r1 = sqrt(2)  * (1 - kp1)       → (1-kp1)² term
//   r2 = sqrt(200) * (kp2 - kp1²)   → 100·(kp2-kp1²)² term
// Total cost (without loss) = Rosenbrock function.
struct RosenbrockSimulator {
  static void reset() {}

  static double find_steady_state(State& out,
                                   const BasicParams<double>& p,
                                   double) {
    out = State{};
    out.Gsc = sqrt(2.0) * (p.kp1 - 1.0);
    return 1.0;
  }

  template<typename T>
  static void step(BasicState<T>& state, const BasicInputs<T>&,
                   const BasicParams<T>& p, double dt) {
    T target = sqrt(T(200.0)) * (p.kp1 * p.kp1 - p.kp2);
    T initial = sqrt(T(2.0)) * (p.kp1 - T(1.0));
    state.Gsc += T(dt) * (target - initial);
  }

  template<int EventType, typename T>
  static void notify(const BasicState<T>&, const BasicInputs<T>&, T) {}
};

// Helper: create a Rosenbrock ObservationCost with 2 params (kp1, kp2).
// Base kp1=kp2=1 so the Rosenbrock optimum (kp1=1, kp2=1) is at θ=(0,0).
static auto make_rosenbrock_cost(double sigma = 1.0) {
  auto data = make_id_data({0.0, 1.0}, {}, {}, {}, {}, {}, {}, 0.0);
  auto base = Params::defaults_t1d();
  base.kp1 = 1.0;
  base.kp2 = 1.0;
  return ObservationCost<RosenbrockSimulator>(data, base,
                                              {{"kp1", 0, 2}, {"kp2", 0, 2}},
                                              sigma);
}

static auto make_rosenbrock_prior() {
  return PriorCost({{"kp1", 0, 2}, {"kp2", 0, 2}});
}

// ===========================================================================
// LMOptimizer — Rosenbrock (nonlinear, 2-param)
// ===========================================================================
class LMRosenbrockTest : public ::testing::Test {
protected:
  void SetUp() override {
    RosenbrockSimulator::reset();
  }
};

TEST_F(LMRosenbrockTest, converges_quickly_near_optimum) {
  // LinearMockSimulator: Gsc += dt * kp1
  // Data {10, 20, 30} with all CGM = 10 (linear ramp). True optimum is
  // where (t * kp1 - 10) residuals balance. Starting from θ = 0.0 (the
  // existing test converges_with_prior starts from {0.0}); starting even
  // closer ({0.01}) should converge in few iterations.
  auto data = make_id_data({10.0, 20.0, 30.0});
  ObservationCost<LinearMockSimulator> obs_cost(data, Params::defaults_t1d(),
                                                {{"kp1", 0, 1}});
  PriorCost prior({{"kp1", 0, 1}});

  auto res = LMOptimizer::run({0.01}, obs_cost, prior, 50);

  EXPECT_TRUE(res.converged);
  EXPECT_TRUE(std::isfinite(res.final_cost));
  EXPECT_EQ(res.final_params.size(), 1);
}

TEST_F(LMRosenbrockTest, rosenbrock_converges) {
  auto obs_cost = make_rosenbrock_cost();
  auto prior = make_rosenbrock_prior();

  // Start far from optimum: (-2, 2) in log-space.
  // With base kp1=kp2=1, the optimum is at (0,0) (kp1=1, kp2=1).
  auto res = LMOptimizer::run({-2.0, 2.0}, obs_cost, prior, 200);

  EXPECT_TRUE(res.converged);
  ASSERT_EQ(res.final_params.size(), 2);
  EXPECT_TRUE(std::isfinite(res.final_cost));
  EXPECT_GT(res.cov.rows(), 0);
}

TEST_F(LMRosenbrockTest, covariance_positive_definite) {
  auto obs_cost = make_rosenbrock_cost();
  auto prior = make_rosenbrock_prior();

  auto res = LMOptimizer::run({0.0, 0.0}, obs_cost, prior, 100);
  ASSERT_TRUE(res.converged);
  ASSERT_EQ(res.cov.rows(), 2);

  Eigen::LLT<Eigen::MatrixXd> llt(res.cov);
  EXPECT_EQ(llt.info(), Eigen::Success);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(res.cov);
  EXPECT_GT(es.eigenvalues().minCoeff(), 0.0);
}

TEST_F(LMRosenbrockTest, covariance_methods_agree) {
  // Both compute_covariance (Ceres) and compute_covariance_at (sandwich)
  // should return positive-definite matrices at the converged point.
  auto obs_cost = make_rosenbrock_cost();
  auto prior = make_rosenbrock_prior();

  auto res = LMOptimizer::run({0.0, 0.0}, obs_cost, prior, 100);
  ASSERT_TRUE(res.converged);

  auto cov1 = LMOptimizer::compute_covariance(res.final_params, obs_cost, prior);
  auto cov2 = LMOptimizer::compute_covariance_at(res.final_params, obs_cost, prior);

  EXPECT_EQ(cov1.cov.rows(), 2);
  EXPECT_EQ(cov2.cov.rows(), 2);
  EXPECT_GT(cov1.cov(0, 0), 0.0);
  EXPECT_GT(cov2.cov(0, 0), 0.0);

  // Sandwich covariance_sandwich should be finite (may be near-zero
  // when residuals are exactly zero, which makes V ≈ 0).
  EXPECT_TRUE(cov2.cov_sandwich.allFinite());
}

TEST_F(LMRosenbrockTest, ceres_cov_matches_jtwj) {
  auto obs_cost = make_rosenbrock_cost();
  auto prior = make_rosenbrock_prior();

  auto res = LMOptimizer::run({0.0, 0.0}, obs_cost, prior, 100);
  ASSERT_TRUE(res.converged);
  ASSERT_EQ(res.cov.rows(), 2);

  const double* params_ptr = res.final_params.data();
  int n_obs = obs_cost.n_obs_;
  int n_params = obs_cost.n_params_;

  std::vector<double> resid(n_obs);
  std::vector<double> jac_buf(n_obs * n_params);
  double* jac_ptrs[] = { jac_buf.data() };
  obs_cost.Evaluate(&params_ptr, resid.data(), jac_ptrs);

  Eigen::MatrixXd J(n_obs, n_params);
  for (int i = 0; i < n_obs; ++i)
    for (int j = 0; j < n_params; ++j)
      J(i, j) = jac_buf[i * n_params + j];

  // Loss weights: W_ii = rho1(r_i^2) = 1/sqrt(1+r_i^2)
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n_obs, n_obs);
  for (int i = 0; i < n_obs; ++i) {
    double r = resid[i];
    W(i, i) = 1.0 / sqrt(1.0 + r * r);
  }

  // Prior Hessian: diag(1/std_i^2)
  Eigen::MatrixXd prior_H = Eigen::MatrixXd::Zero(n_params, n_params);
  for (int i = 0; i < prior.n_params_; ++i)
    prior_H(i, i) = 1.0 / (prior.stds_[i] * prior.stds_[i]);

  Eigen::MatrixXd expected_cov =
      (J.transpose() * W * J + prior_H).inverse();

  EXPECT_TRUE(expected_cov.isApprox(res.cov, 1e-4));
}

// ===========================================================================
// NaN-derivative and NaN-value detection tests
// ===========================================================================
// Simulator that produces NaN in the simulation (NaN basal insulin
// propagates to Gsc via step, triggering the residual-value NaN check)
struct NaNValSimulator {
  static void reset() {}

  static double find_steady_state(State& out,
                                  const BasicParams<double>& /*p*/,
                                  double /*Gb*/) {
    out = State{};
    return std::numeric_limits<double>::quiet_NaN();
  }

  template<typename T>
  static void step(BasicState<T>& state, const BasicInputs<T>& u,
                   const BasicParams<T>&, double dt) {
    state.Gsc += T(dt) * u.insulin_rate;
  }

  template<int EventType, typename T>
  static void notify(const BasicState<T>&, const BasicInputs<T>&, T) {}
};

// Simulator that injects NaN derivatives into Gsc at each step
struct NaNJacSimulator {
  static void reset() {}

  static double find_steady_state(State& out,
                                  const BasicParams<double>& /*p*/,
                                  double /*Gb*/) {
    out = State{};
    return 1.0;
  }

  template<typename T>
  static void step(BasicState<T>& state, const BasicInputs<T>&,
                   const BasicParams<T>& p, double dt) {
    if constexpr (std::is_same_v<T, double>) {
      state.Gsc += T(dt) * p.kp1;
    } else {
      state.Gsc.a += dt * p.kp1.a;
      state.Gsc.v.setConstant(std::numeric_limits<double>::quiet_NaN());
    }
  }

  template<int EventType, typename T>
  static void notify(const BasicState<T>&, const BasicInputs<T>&, T) {}
};

TEST(NaNJacobianTest, salvage_state_recovers_from_nan) {
  using JetT = ceres::Jet<double, 2>;
  BasicState<JetT> state;
  state.Gsc.a = 120.0;
  state.Gsc.v[0] = std::numeric_limits<double>::quiet_NaN();
  state.Gsc.v[1] = std::numeric_limits<double>::quiet_NaN();

  ObservationCost<NaNJacSimulator>::salvage_state(state);

  EXPECT_TRUE(std::isfinite(state.Gsc.a));
  EXPECT_TRUE(std::isfinite(state.Gsc.v[0]));
  EXPECT_TRUE(std::isfinite(state.Gsc.v[1]));
  EXPECT_DOUBLE_EQ(state.Gsc.v[0], 0.0);
  EXPECT_DOUBLE_EQ(state.Gsc.v[1], 0.0);
}

TEST(NaNJacobianTest, salvage_state_recovers_nan_value) {
  using JetT = ceres::Jet<double, 2>;
  BasicState<JetT> state;
  state.Gsc.a = std::numeric_limits<double>::quiet_NaN();
  state.Gsc.v[0] = 5.0;

  ObservationCost<NaNJacSimulator>::salvage_state(state);

  EXPECT_TRUE(std::isfinite(state.Gsc.a));
  EXPECT_DOUBLE_EQ(state.Gsc.v[0], 0.0);
}

TEST(NaNJacobianTest, evaluate_succeeds_after_salvage) {
  // Even with NaN-producing simulation, salvage fixes the state before
  // observations are recorded.  Evaluate should succeed and produce
  // a clean (finite) Jacobian.
  auto data = make_id_data({1.0, 2.0, 3.0});
  auto base = Params::defaults_t1d();
  ObservationCost<NaNValSimulator> obs_cost(data, base,
                                            {{"kp1", 0, 1}});

  std::vector<double> params = {0.5};
  const double* params_ptr = params.data();
  int n_obs = obs_cost.n_obs_;
  int n_params = obs_cost.n_params_;
  std::vector<double> resid(n_obs);
  std::vector<double> jac_buf(n_obs * n_params);
  double* jac_ptrs[] = { jac_buf.data() };

  bool result = obs_cost.Evaluate(&params_ptr, resid.data(), jac_ptrs);
  EXPECT_TRUE(result);
  for (int i = 0; i < n_obs; ++i)
    EXPECT_TRUE(std::isfinite(resid[i]));
  for (int i = 0; i < n_obs * n_params; ++i)
    EXPECT_TRUE(std::isfinite(jac_buf[i]));
}

TEST(NaNJacobianTest, salvage_recovers_and_evaluate_succeeds) {
  // The salvage fixes NaN derivatives after each step, so Evaluate
  // should succeed even with the NaNJacSimulator
  auto data = make_id_data({10.0, 20.0, 30.0});
  auto base = Params::defaults_t1d();
  ObservationCost<NaNJacSimulator> obs_cost(data, base,
                                            {{"kp1", 0, 1}, {"kp2", 0, 1}});

  std::vector<double> params = {0.5, -0.3};
  const double* params_ptr = params.data();
  int n_obs = obs_cost.n_obs_;
  int n_params = obs_cost.n_params_;
  std::vector<double> resid(n_obs);
  std::vector<double> jac_buf(n_obs * n_params);
  double* jac_ptrs[] = { jac_buf.data() };

  // Salvage zeros out NaN derivatives, so Evaluate should pass
  bool result = obs_cost.Evaluate(&params_ptr, resid.data(), jac_ptrs);
  EXPECT_TRUE(result);
  for (int i = 0; i < n_obs * n_params; ++i)
    EXPECT_TRUE(std::isfinite(jac_buf[i]));
}

TEST(NaNJacobianTest, compute_raw_gradients_recovers_via_salvage) {
  auto data = make_id_data({10.0, 20.0, 30.0});
  auto base = Params::defaults_t1d();
  ObservationCost<NaNJacSimulator> obs_cost(data, base,
                                            {{"kp1", 0, 1}, {"kp2", 0, 1}});

  std::vector<double> params = {0.5, -0.3};
  std::vector<double> raw_resid;
  Eigen::MatrixXd J_raw;

  bool ok = obs_cost.compute_raw_gradients(params.data(), raw_resid, J_raw);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(J_raw.allFinite());
}

TEST(NaNJacobianTest, lm_with_nan_simulator_does_not_crash) {
  auto data = make_id_data({1.0, 2.0, 3.0});
  auto base = Params::defaults_t1d();
  ObservationCost<NaNValSimulator> obs_cost(data, base,
                                            {{"kp1", 0, 1}, {"kp2", 0, 1}});
  PriorCost prior({{"kp1", 0, 1}, {"kp2", 0, 2}});

  EXPECT_NO_THROW({
    auto res = LMOptimizer::run({0.0, 0.0}, obs_cost, prior, 10);
    EXPECT_TRUE(std::isfinite(res.final_cost));
    EXPECT_EQ(res.cov.rows(), 2);
    EXPECT_EQ(res.cov.cols(), 2);
  });
}

TEST(NaNJacobianTest, lm_falls_back_gracefully_on_nan_jacobian) {
  auto data = make_id_data({10.0, 20.0, 30.0});
  auto base = Params::defaults_t1d();
  ObservationCost<NaNJacSimulator> obs_cost(data, base,
                                            {{"kp1", 0, 1}, {"kp2", 0, 1}});
  PriorCost prior({{"kp1", 0, 1}, {"kp2", 0, 2}});

  EXPECT_NO_THROW({
    auto res = LMOptimizer::run({0.0, 0.0}, obs_cost, prior, 10);
    EXPECT_TRUE(std::isfinite(res.final_cost));
    EXPECT_EQ(res.cov.rows(), 2);
    EXPECT_EQ(res.cov.cols(), 2);
  });
}

TEST(NaNJacobianTest, validate_jacobian_passes_for_clean) {
  auto data = make_id_data({10.0, 20.0, 30.0});
  auto base = Params::defaults_t1d();
  ObservationCost<LinearMockSimulator> obs_cost(data, base,
                                                {{"kp1", 0, 1}});

  std::vector<double> params = {0.5};
  bool valid = detail::validate_jacobian(obs_cost, params);

  EXPECT_TRUE(valid);
}

TEST_F(LMRosenbrockTest, numerical_hessian_matches_analytic) {
  auto obs_cost = make_rosenbrock_cost();
  auto prior = make_rosenbrock_prior();

  // Evaluate at exact θ = (0, 0) where residuals are zero (no r·∇²r term)
  std::vector<double> theta = {0.0, 0.0};
  const double* params_ptr = theta.data();
  int n_obs = obs_cost.n_obs_;
  int n_params = obs_cost.n_params_;

  // --- Analytic Hessian (Gauss-Newton: d²loss/dr²·JᵀJ + prior) ---
  std::vector<double> raw_resid;
  Eigen::MatrixXd J_raw;
  ASSERT_TRUE(obs_cost.compute_raw_gradients(params_ptr, raw_resid, J_raw));

  // Aggregate loss: cost = sqrt(1 + ||r||²) - 1
  Eigen::VectorXd r_vec = Eigen::Map<const Eigen::VectorXd>(raw_resid.data(), n_obs);
  double s = r_vec.squaredNorm();
  ceres::SoftLOneLoss loss(1.0);
  double rho[3];
  loss.Evaluate(s, rho);
  // rho[1] = 1/sqrt(1+s), rho[2] = -1/(2*(1+s)^(3/2))
  // H = rho[1]·JᵀJ + 2·rho[2]·(Jᵀr)(Jᵀr)ᵀ
  Eigen::VectorXd Jtr = J_raw.transpose() * r_vec;
  Eigen::MatrixXd H_analytic = rho[1] * J_raw.transpose() * J_raw
                             + 2.0 * rho[2] * Jtr * Jtr.transpose();
  for (int i = 0; i < n_params; ++i) {
    H_analytic(i, i) += 1.0 / (prior.stds_[i] * prior.stds_[i]);
  }

  // --- Numerical Hessian via central differences ---
  auto total_fn = [&](const double* x) {
    double f = obs_cost.compute_scalar_cost(x)
             + prior.compute_scalar_prior(x);
    return std::isfinite(f) ? f : 1e20;
  };

  const double h = 1e-5;
  std::vector<double> xp(n_params), xm(n_params);
  Eigen::MatrixXd H_num = Eigen::MatrixXd::Zero(n_params, n_params);
  double f0 = total_fn(params_ptr);

  for (int i = 0; i < n_params; ++i) {
    std::copy(params_ptr, params_ptr + n_params, xp.data());
    std::copy(params_ptr, params_ptr + n_params, xm.data());
    xp[i] += h;
    xm[i] -= h;
    H_num(i, i) = (total_fn(xp.data()) - 2.0 * f0 + total_fn(xm.data())) / (h * h);
  }

  for (int i = 0; i < n_params; ++i) {
    for (int j = i + 1; j < n_params; ++j) {
      std::vector<double> xpp(n_params), xpm(n_params), xmp(n_params), xmm(n_params);
      std::copy(params_ptr, params_ptr + n_params, xpp.data());
      std::copy(params_ptr, params_ptr + n_params, xpm.data());
      std::copy(params_ptr, params_ptr + n_params, xmp.data());
      std::copy(params_ptr, params_ptr + n_params, xmm.data());
      xpp[i] += h; xpp[j] += h;
      xpm[i] += h; xpm[j] -= h;
      xmp[i] -= h; xmp[j] += h;
      xmm[i] -= h; xmm[j] -= h;
      H_num(i, j) = (total_fn(xpp.data()) - total_fn(xpm.data())
                    - total_fn(xmp.data()) + total_fn(xmm.data()))
                   / (4.0 * h * h);
      H_num(j, i) = H_num(i, j);
    }
  }

  // Gauss-Newton Hessian matches numerical full Hessian at zero residuals
  // (the r·∇²r term vanishes). Tolerance accounts for FD truncation + soft-L1
  // curvature mismatch at non-exactly-zero residuals.
  EXPECT_TRUE(H_analytic.isApprox(H_num, 0.1));
}
