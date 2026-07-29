#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include <ceres/ceres.h>

#include "uva_padova/cost_function.hpp"
#include "mocks/mock_simulator.h"
#include "test_data/synthetic_data.h"

using namespace uva_padova;

// ---------------------------------------------------------------------------
//  Death test: n_params_ > kMaxJetDim → std::abort
// ---------------------------------------------------------------------------
TEST(ObservationCostDeathTest, too_many_params) {
  std::vector<ParamDescriptor> descs(50, ParamDescriptor{"x"});
  IdentificationData d;
  d.finalize();
  EXPECT_DEATH({ ObservationCost<MockSimulator> cost(d, Params{}, descs); }, "");
}

// ---------------------------------------------------------------------------
//  Empty data: no CGM observations → total_min = 0, Evaluate OK
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, empty_data) {
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  auto data = make_id_data({});
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  EXPECT_EQ(cost.n_obs_, 0);
  EXPECT_EQ(data.total_min(), 0);
  EXPECT_EQ(cost.num_residuals(), 0);

  std::vector<double> theta = {0.0};
  std::vector<double> residuals;
  const double* ptr = theta.data();
  EXPECT_TRUE(cost.Evaluate(&ptr, nullptr, nullptr));
  EXPECT_EQ(cost.eval_count(), 1);
}

// ---------------------------------------------------------------------------
//  Basic Evaluate: no Jacobians requested (double path)
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, evaluate_no_jacobian) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 100.0;
  MockSimulator::basal_insulin = 1.0;

  auto data = make_id_data({0.0, 30.0, 60.0}, {}, {}, {}, {}, {}, {}, 120.0);
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta = {0.0};
  std::vector<double> residuals(cost.n_obs_);
  const double* ptr = theta.data();
  EXPECT_TRUE(cost.Evaluate(&ptr, residuals.data(), nullptr));

  // Since Gsc stays at 100.0 (identity dyn) and obs = 120.0:
  // r = (100 - 120) / 2.9 = -6.89655...
  double expected = (100.0 - 120.0) / 2.9;
  for (int i = 0; i < cost.n_obs_; ++i)
    EXPECT_NEAR(residuals[i], expected, 1e-10);

  // Mock should have been called total_min times
  EXPECT_EQ(MockSimulator::step_count, 60);
}

// ---------------------------------------------------------------------------
//  Evaluate with Jacobians: Jet path
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, evaluate_with_jacobian) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 100.0;
  MockSimulator::basal_insulin = 1.0;

  auto data = make_id_data({0.0, 10.0}, {}, {}, {}, {}, {}, {}, 120.0);
  std::vector<ParamDescriptor> descs = {{"kp1"}, {"kp2"}};
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta = {0.0, 0.0};
  std::vector<double> residuals(cost.n_obs_);
  std::vector<double> J_buf(cost.n_obs_ * cost.n_params_);
  double* ptrs[] = {J_buf.data()};
  const double* tptr = theta.data();
  EXPECT_TRUE(cost.Evaluate(&tptr, residuals.data(), ptrs));

  // With MockSimulator, steady state derivatives are 0 (fixed SS)
  // and step is identity for Jet, so Gsc derivative = 0
  for (int i = 0; i < cost.n_obs_; ++i) {
    for (int j = 0; j < cost.n_params_; ++j) {
      EXPECT_DOUBLE_EQ(J_buf[i * cost.n_params_ + j], 0.0);
    }
  }
}

// ---------------------------------------------------------------------------
//  compute_scalar_cost: matches manual formula
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, scalar_cost) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 100.0;

  auto data = make_id_data({0.0, 10.0}, {}, {}, {}, {}, {}, {}, 120.0);
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta = {0.0};

  // r = (100 - 120) / 2.9
  double r = (100.0 - 120.0) / 2.9;
  double per_element = std::sqrt(1.0 + r * r) - 1.0;
  double expected = per_element * cost.n_obs_;

  EXPECT_NEAR(cost.compute_scalar_cost(theta.data()), expected, 1e-10);
}

// ---------------------------------------------------------------------------
//  scalar_cost is zero when all residuals are zero
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, scalar_cost_zero) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 120.0;  // Matches cgm_value_fill in make_id_data

  auto data = make_id_data({0.0, 5.0, 10.0}, {}, {}, {}, {}, {}, {}, 120.0);
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta = {0.0};
  EXPECT_NEAR(cost.compute_scalar_cost(theta.data()), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
//  scalar_cost_per_element_matches_ceres: per-observation Ceres problem
//  produces same final_cost as compute_scalar_cost
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, scalar_cost_per_element_matches_ceres) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 100.0;
  MockSimulator::basal_insulin = 1.0;

  auto data = make_id_data({0.0, 10.0}, {}, {}, {}, {}, {}, {}, 120.0);
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<MockSimulator> obs_cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta = {0.0};
  double scalar = obs_cost.compute_scalar_cost(theta.data());
  EXPECT_TRUE(std::isfinite(scalar));

  auto cache = std::make_shared<SimulationCache>();
  ceres::Problem problem;
  for (int i = 0; i < obs_cost.n_obs_; ++i) {
    problem.AddResidualBlock(
        new PerObservationCost<MockSimulator>(&obs_cost, i, cache),
        new ceres::SoftLOneLoss(1.0), theta.data());
  }

  ceres::Problem::EvaluateOptions eval_opts;
  double ceres_cost = 0.0;
  problem.Evaluate(eval_opts, &ceres_cost, nullptr, nullptr, nullptr);

  EXPECT_NEAR(ceres_cost, scalar, 1e-10);
}

// ---------------------------------------------------------------------------
//  simulate_observations: basic identity dynamics
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, simulate_identity) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 100.0;
  MockSimulator::basal_insulin = 1.0;

  auto data = make_id_data({0.0, 30.0, 60.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta = {0.0};
  auto cgm = cost.simulate_observations<double>(theta.data());

  ASSERT_EQ(cgm.size(), 3);
  EXPECT_NEAR(cgm[0], 100.0, 1e-10);
  EXPECT_NEAR(cgm[1], 100.0, 1e-10);
  EXPECT_NEAR(cgm[2], 100.0, 1e-10);
}

// ---------------------------------------------------------------------------
//  simulate_observations: single meal event at t=30
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, simulate_with_meal) {
  MockSimulator::reset();
  MockSimulator::ss_state = State{};
  MockSimulator::ss_state.Gsc = 100.0;
  MockSimulator::basal_insulin = 1.0;

  // When mealD > 0, step adds a bit to Gsc so we can detect the meal
  // using the step_delta callback.  But step_delta only gets params,
  // not state.  Instead, we count steps — the meal doesn't affect step
  // count, but we verify that simulation runs without error and produces
  // the right number of outputs at the right times.
  MockSimulator::step_delta = [](const BasicParams<double>&) { return 0.1; };

  // Observations at t=0 (before meal), t=30 (same moment as meal),
  // t=60 (after meal).  According to the simulation loop:
  //   - t=0:  record Gsc (100.0), then step (→100.1)
  //   - t=1..29: step each minute
  //   - t=30: apply meal (Qsto1 += 50000, mealD=50000), record Gsc (after 29 steps
  //            of step_delta = 100.0 + 29*0.1 = 102.9, then step → 103.0)
  // Wait — step order: at t=30, first apply meal, THEN record CGM, THEN step
  // So recorded Gsc at t=30 is BEFORE the step at this iteration.
  // After 29 steps (t=0..28): Gsc = 100.0 + 29*0.1 = 102.9
  // At t=29: step → Gsc = 103.0
  // At t=30: apply meal, then record Gsc = 103.0, then step → 103.1
  
  auto data = make_id_data({0.0, 30.0, 60.0},
                           {30.0}, {50000.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta = {0.0};
  auto cgm = cost.simulate_observations<double>(theta.data());

  ASSERT_EQ(cgm.size(), 3);
  // t=0: recorded before any step
  EXPECT_NEAR(cgm[0], 100.0, 1e-10);
  // t=30: after 30 steps (t=0..29) = 100.0 + 30*0.1 = 103.0
  // Actually at t=0: after recording, step → 100.1
  // t=1: step → 100.2
  // ...
  // t=29: record at t=0 doesn't happen again; step at t=29: 100.0 + 30*0.1 = 103.0
  // Let me recount: steps happen at t=0,1,2,...,59 (total 60 steps if total_min=60)
  // At t=0: record (100.0), then step (→100.1) — step 1
  // At t=1: step (→100.2) — step 2
  // ...
  // At t=29: step (→103.0) — step 30
  // At t=30: record (103.0), then step (→103.1) — step 31
  // At t=59: step (→109.0) — step 60
  // At t=60: record (109.0), loop ends — no step at t=60 since t < total_min = false
  
  // Actually wait, the loop is: for t = 0..total_min inclusive.
  // Step is: if (t < data_.total_min()) Simulator::step(...)
  // So steps happen at t=0,1,...,total_min-1 = 0..59 (60 steps)
  
  // By t=30 (before step at t=30), we've done steps at t=0..29 = 30 steps
  // Gsc = 100.0 + 30*0.1 = 103.0
  EXPECT_NEAR(cgm[1], 103.0, 1e-10);
  
  // By t=60 (before step at t=60 — but step doesn't happen since t=60 == total_min)
  // Steps done: t=0..59 = 60 steps
  // Gsc = 100.0 + 60*0.1 = 106.0
  // At t=60: record, loop ends
  EXPECT_NEAR(cgm[2], 106.0, 1e-10);
}

// ---------------------------------------------------------------------------
//  Time rounding: cgm_time with fractional minutes
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, time_rounding) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 100.0;
  MockSimulator::basal_insulin = 1.0;
  MockSimulator::step_delta = [](const BasicParams<double>&) { return 1.0; };

  // Times: 0.49 rounds to 0, 1.51 rounds to 2
  auto data = make_id_data({0.49, 1.51});
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta = {0.0};
  auto cgm = cost.simulate_observations<double>(theta.data());

  ASSERT_EQ(cgm.size(), 2);
  // t=0: record at 0.49, rounded to 0 → 0 steps yet
  EXPECT_NEAR(cgm[0], 100.0, 1e-10);
  // t=2: record at 1.51, rounded to 2 → after 2 steps (t=0,1)
  // Gsc = 100.0 + 2*1.0 = 102.0
  EXPECT_NEAR(cgm[1], 102.0, 1e-10);
}

// ---------------------------------------------------------------------------
//  Observation outside simulation window: obs_t beyond total_min
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, observation_outside_window) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 100.0;
  MockSimulator::step_delta = [](const BasicParams<double>&) { return 1.0; };

  // total_min is derived from cgm_time.back(): 20 minutes
  auto data = make_id_data({5.0, 20.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  EXPECT_EQ(data.total_min(), 20);

  std::vector<double> theta = {0.0};
  auto cgm = cost.simulate_observations<double>(theta.data());
  // Both observations should be recorded since total_min extends to cover
  // the max of cgm_time.back() (via finalize).
  ASSERT_EQ(cgm.size(), 2);
  // After 5 steps (t=0..4): Gsc = 100.0 + 5*1.0 = 105.0
  EXPECT_NEAR(cgm[0], 105.0, 1e-10);
  // After 20 steps (t=0..19): Gsc = 100.0 + 20*1.0 = 120.0
  EXPECT_NEAR(cgm[1], 120.0, 1e-10);
}

// ---------------------------------------------------------------------------
//  Gradient check: analytical Jet Jacobian vs finite differences
//  Uses LinearMockSimulator for predictable differentiable dynamics
// ---------------------------------------------------------------------------
TEST(ObservationCostGradientTest, finite_difference_check) {
  LinearMockSimulator::reset();
  LinearMockSimulator::ss_state.Gsc = 100.0;
  LinearMockSimulator::basal_insulin = 1.0;

  auto data = make_id_data({0.0, 15.0, 30.0},
                           {}, {}, {}, {}, {}, {}, 120.0);

  // We'll adjust two params: kp1 and VG (to have multiple active params)
  std::vector<ParamDescriptor> descs = {{"kp1"}, {"VG"}};
  Params base = Params::defaults_t1d();

  ObservationCost<LinearMockSimulator> cost(data, base, descs);

  std::vector<double> theta = {0.1, -0.05};  // arbitrary test point

  // --- Analytical Jacobian via Evaluate ---
  std::vector<double> residuals(cost.n_obs_);
  std::vector<double> J_ana(cost.n_obs_ * cost.n_params_);
  double* J_ptrs[] = {J_ana.data()};
  const double* tptr = theta.data();
  ASSERT_TRUE(cost.Evaluate(&tptr, residuals.data(), J_ptrs));

  // --- Finite-difference Jacobian (central differences, O(eps^2)) ---
  const double eps = 1e-6;
  auto fd_sim = [&](const double* th) {
    return cost.simulate_observations<double>(th);
  };

  std::vector<double> J_fd(cost.n_obs_ * cost.n_params_, 0.0);

  for (int j = 0; j < cost.n_params_; ++j) {
    std::vector<double> th_lo = theta;
    th_lo[j] -= eps;
    std::vector<double> th_hi = theta;
    th_hi[j] += eps;
    auto cgm_lo = fd_sim(th_lo.data());
    auto cgm_hi = fd_sim(th_hi.data());

    for (int i = 0; i < cost.n_obs_; ++i) {
      // Central difference: dr/dθ ≈ (r(θ+ε) − r(θ−ε)) / (2ε)
      double r_lo = (cgm_lo[i] - data.cgm_value[i]) / cost.sigma_;
      double r_hi = (cgm_hi[i] - data.cgm_value[i]) / cost.sigma_;
      J_fd[i * cost.n_params_ + j] = (r_hi - r_lo) / (2.0 * eps);
    }
  }

  // Compare
  const double tol = 1e-6;
  for (int i = 0; i < cost.n_obs_; ++i) {
    for (int j = 0; j < cost.n_params_; ++j) {
      EXPECT_NEAR(J_ana[i * cost.n_params_ + j],
                  J_fd[i * cost.n_params_ + j], tol)
          << " at obs=" << i << " param=" << j;
    }
  }
}

// ---------------------------------------------------------------------------
//  Gradient convergence: Jet Jacobian vs finite differences at multiple ε.
//  Uses a single active param (kp1) so all Jacobian entries are meaningful.
//  Verifies O(ε²) convergence of central differences for ε where truncation
//  error dominates roundoff (ε ≥ 1e-4 with double precision).
// ---------------------------------------------------------------------------
TEST(ObservationCostGradientTest, gradient_convergence) {
  LinearMockSimulator::reset();
  LinearMockSimulator::ss_state.Gsc = 100.0;
  LinearMockSimulator::basal_insulin = 1.0;

  auto data = make_id_data({0.0, 15.0, 30.0},
                           {}, {}, {}, {}, {}, {}, 120.0);
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<LinearMockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta = {0.1};
  const int n_obs = cost.n_obs_;
  const int n_params = cost.n_params_;

  // Analytical Jacobian via Evaluate (Jet)
  std::vector<double> residuals(n_obs);
  std::vector<double> J_ana(n_obs * n_params);
  double* J_ptrs[] = {J_ana.data()};
  const double* tptr = theta.data();
  ASSERT_TRUE(cost.Evaluate(&tptr, residuals.data(), J_ptrs));

  // Finite-difference at multiple epsilons
  auto fd_J = [&](double eps) {
    std::vector<double> J_fd(n_obs * n_params, 0.0);
    for (int j = 0; j < n_params; ++j) {
      std::vector<double> th_lo = theta, th_hi = theta;
      th_lo[j] -= eps;
      th_hi[j] += eps;
      auto cgm_lo = cost.simulate_observations<double>(th_lo.data());
      auto cgm_hi = cost.simulate_observations<double>(th_hi.data());
      for (int i = 0; i < n_obs; ++i) {
        double r_lo = (cgm_lo[i] - data.cgm_value[i]) / cost.sigma_;
        double r_hi = (cgm_hi[i] - data.cgm_value[i]) / cost.sigma_;
        J_fd[i * n_params + j] = (r_hi - r_lo) / (2.0 * eps);
      }
    }
    return J_fd;
  };

  // ε=1e-3 and ε=1e-4 are in truncation-dominated regime (O(ε²)).
  // ε=1e-5 is near the optimal ε ≈ 3e-6 for this model, so roundoff
  // contaminates the truncation error — we skip it.
  auto J_1e3 = fd_J(1e-3);
  auto J_1e4 = fd_J(1e-4);
  double err_1e3 = 0.0, err_1e4 = 0.0;
  for (int i = 0; i < n_obs * n_params; ++i) {
    err_1e3 = std::max(err_1e3, std::abs(J_ana[i] - J_1e3[i]));
    err_1e4 = std::max(err_1e4, std::abs(J_ana[i] - J_1e4[i]));
  }
  ASSERT_TRUE(std::isfinite(err_1e3));
  ASSERT_TRUE(std::isfinite(err_1e4));

  // Central difference error ∝ ε², so 10× smaller ε → 100× smaller error
  double ratio = err_1e3 / err_1e4;
  ASSERT_TRUE(ratio > 50.0);
}

// ---------------------------------------------------------------------------
//  Param sensitivity: verify that kp1 always affects Gsc.
//  Guards against accidentally choosing a descriptor whose parameter
//  has zero effect on the simulated CGM trajectory.
// ---------------------------------------------------------------------------
TEST(ObservationCostGradientTest, param_sensitivity) {
  LinearMockSimulator::reset();
  LinearMockSimulator::ss_state.Gsc = 100.0;
  LinearMockSimulator::basal_insulin = 1.0;

  auto data = make_id_data({0.0, 15.0, 30.0});
  ObservationCost<LinearMockSimulator> cost(data, Params::defaults_t1d(), {{"kp1"}});

  std::vector<double> theta_base = {0.0};
  std::vector<double> theta_pert = {1e-4};
  auto cgm_base = cost.simulate_observations<double>(theta_base.data());
  auto cgm_pert = cost.simulate_observations<double>(theta_pert.data());

  ASSERT_EQ(cgm_base.size(), cgm_pert.size());
  // Observations after at least one step (t > 0) must differ.
  // t=0 is recorded before any step, so it's always ss_state.Gsc.
  for (int i = 1; i < cgm_base.size(); ++i) {
    double diff = cgm_pert[i] - cgm_base[i];
    ASSERT_TRUE(std::isfinite(diff));
    ASSERT_NE(diff, 0.0);
  }
}

// ---------------------------------------------------------------------------
//  compute_raw_gradients returns matching Jacobian
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, raw_gradients_match_evaluate) {
  LinearMockSimulator::reset();
  LinearMockSimulator::ss_state.Gsc = 100.0;
  LinearMockSimulator::basal_insulin = 1.0;

  auto data = make_id_data({0.0, 10.0, 20.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}, {"Vm0"}};
  ObservationCost<LinearMockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta = {0.2, -0.1};

  // Evaluate with Jacobian
  std::vector<double> r1(cost.n_obs_);
  std::vector<double> J1(cost.n_obs_ * cost.n_params_);
  double* J1p[] = {J1.data()};
  const double* tp = theta.data();
  cost.Evaluate(&tp, r1.data(), J1p);

  // compute_raw_gradients
  std::vector<double> r2;
  Eigen::MatrixXd J2;
  cost.compute_raw_gradients(theta.data(), r2, J2);

  ASSERT_EQ(r2.size(), r1.size());
  for (int i = 0; i < cost.n_obs_; ++i)
    EXPECT_NEAR(r2[i], r1[i], 1e-12);

  for (int i = 0; i < cost.n_obs_; ++i)
    for (int j = 0; j < cost.n_params_; ++j)
      EXPECT_NEAR(J2(i, j), J1[i * cost.n_params_ + j], 1e-12);
}

// ---------------------------------------------------------------------------
//  Insulin events: u.insulin_rate increases at event times
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, simulate_with_insulin_events) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 100.0;
  MockSimulator::basal_insulin = 1.0;

  // Insulin events: +5 at t=10, +3 at t=20
  auto data = make_id_data({0.0, 30.0}, {}, {},
                           {10.0, 20.0}, {5.0, 3.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> observed_rates;
  MockSimulator::step_observer =
      [&](const BasicState<double>&, const BasicInputs<double>& u) {
        observed_rates.push_back(u.insulin_rate);
      };

  std::vector<double> theta = {0.0};
  cost.simulate_observations<double>(theta.data());

  // 30 steps (t=0..29). Observer fires after step_count incremented.
  // step_count-1 = t (step_count starts at 1 on first call).
  // Event at t=10 is applied BEFORE step at t=10 → observed in entry 10.
  //   t=0..9:   rate = 1.0 (basal only)
  //   t=10..19: rate = 6.0 (basal + 5)
  //   t=20..29: rate = 9.0 (basal + 5 + 3)
  ASSERT_EQ(observed_rates.size(), 30);
  for (int i = 0; i < 10; ++i)
    EXPECT_NEAR(observed_rates[i], 1.0, 1e-10) << "t=" << i;
  for (int i = 10; i < 20; ++i)
    EXPECT_NEAR(observed_rates[i], 6.0, 1e-10) << "t=" << i;
  for (int i = 20; i < 30; ++i)
    EXPECT_NEAR(observed_rates[i], 9.0, 1e-10) << "t=" << i;
}

// ---------------------------------------------------------------------------
//  Bolus events: state.Isc1 increases at event times
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, simulate_with_bolus_events) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 100.0;
  MockSimulator::basal_insulin = 1.0;

  // Bolus event: +200 at t=15
  auto data = make_id_data({0.0, 20.0}, {}, {},
                           {}, {},
                           {15.0}, {200.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> observed_isc1;
  MockSimulator::step_observer =
      [&](const BasicState<double>& s, const BasicInputs<double>&) {
        observed_isc1.push_back(s.Isc1);
      };

  std::vector<double> theta = {0.0};
  cost.simulate_observations<double>(theta.data());

  // 20 steps (t=0..19).
  // Event at t=15 is applied BEFORE step at t=15 → observed in entry 15.
  //   t=0..14:  Isc1 = 0.0 (no bolus yet)
  //   t=15..19: Isc1 = 200.0
  ASSERT_EQ(observed_isc1.size(), 20);
  for (int i = 0; i < 15; ++i)
    EXPECT_NEAR(observed_isc1[i], 0.0, 1e-10) << "t=" << i;
  for (int i = 15; i < 20; ++i)
    EXPECT_NEAR(observed_isc1[i], 200.0, 1e-10) << "t=" << i;
}

// ---------------------------------------------------------------------------
//  Simultaneous insulin, bolus, and meal at the same time step.
//  Verifies event_log order and aggregate state after all three.
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, simulate_simultaneous_events) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 100.0;
  MockSimulator::basal_insulin = 1.0;

  // All three events at t=15
  auto data = make_id_data({0.0, 30.0},
                           {15.0}, {50000.0},
                           {15.0}, {2.0},
                           {15.0}, {100.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> obs_rate, obs_isc1, obs_qsto1;
  MockSimulator::step_observer =
      [&](const BasicState<double>& s, const BasicInputs<double>& u) {
        obs_rate.push_back(u.insulin_rate);
        obs_isc1.push_back(s.Isc1);
        obs_qsto1.push_back(s.Qsto1);
      };

  std::vector<double> theta = {0.0};
  auto cgm = cost.simulate_observations<double>(theta.data());

  // --- event_log: all three fired, in order, at t=15 ---
  ASSERT_EQ(MockSimulator::event_log.size(), 3);
  EXPECT_EQ(MockSimulator::event_log[0], std::pair(0, 15));
  EXPECT_EQ(MockSimulator::event_log[1], std::pair(1, 15));
  EXPECT_EQ(MockSimulator::event_log[2], std::pair(2, 15));

  // --- step_observer: 30 steps (t=0..29) ---
  ASSERT_EQ(obs_rate.size(), 30);
  // t=0..14: basal only
  for (int i = 0; i < 15; ++i) {
    EXPECT_NEAR(obs_rate[i], 1.0, 1e-10) << "rate t=" << i;
    EXPECT_NEAR(obs_isc1[i], 0.0, 1e-10) << "isc1 t=" << i;
    EXPECT_NEAR(obs_qsto1[i], 0.0, 1e-10) << "qsto1 t=" << i;
  }
  // t=15..29: all three events applied
  for (int i = 15; i < 30; ++i) {
    EXPECT_NEAR(obs_rate[i], 3.0, 1e-10) << "rate t=" << i;
    EXPECT_NEAR(obs_isc1[i], 100.0, 1e-10) << "isc1 t=" << i;
    EXPECT_NEAR(obs_qsto1[i], 50000.0, 1e-10) << "qsto1 t=" << i;
  }

  // --- CGM outputs ---
  ASSERT_EQ(cgm.size(), 2);
  EXPECT_NEAR(cgm[0], 100.0, 1e-10);
  // t=30: after 30 steps, Gsc unchanged (mock identity dynamics)
  EXPECT_NEAR(cgm[1], 100.0, 1e-10);
}

// ---------------------------------------------------------------------------
//  Events past total_min() are silently ignored.
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, simulate_events_past_total_min) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 100.0;
  MockSimulator::basal_insulin = 1.0;

  // total_min is derived from cgm_time.back(): 20 minutes.
  // One insulin event at t=10 (within window), one at t=100 (past).
  // One bolus event at t=200 (past).
  auto data = make_id_data({0.0, 20.0}, {}, {},
                           {10.0, 100.0}, {1.0, 999.0},
                           {200.0}, {500.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> obs_rate, obs_isc1;
  MockSimulator::step_observer =
      [&](const BasicState<double>& s, const BasicInputs<double>& u) {
        obs_rate.push_back(u.insulin_rate);
        obs_isc1.push_back(s.Isc1);
      };

  std::vector<double> theta = {0.0};
  cost.simulate_observations<double>(theta.data());

  // --- event_log: only the insulin at t=10 fired ---
  ASSERT_EQ(MockSimulator::event_log.size(), 1);
  EXPECT_EQ(MockSimulator::event_log[0], std::pair(0, 10));

  // --- step_observer: 20 steps ---
  ASSERT_EQ(obs_rate.size(), 20);
  // t=0..9: basal only
  for (int i = 0; i < 10; ++i)
    EXPECT_NEAR(obs_rate[i], 1.0, 1e-10) << "rate t=" << i;
  // t=10..19: basal + 1.0
  for (int i = 10; i < 20; ++i)
    EXPECT_NEAR(obs_rate[i], 2.0, 1e-10) << "rate t=" << i;
  // Isc1 never touched (bolus at 200 is past total_min)
  for (int i = 0; i < 20; ++i)
    EXPECT_NEAR(obs_isc1[i], 0.0, 1e-10) << "isc1 t=" << i;
}

// ---------------------------------------------------------------------------
//  sigma scales Jacobian by 1/sigma and changes SoftLOne cost nonlinearly.
// ---------------------------------------------------------------------------
TEST(ObservationCostTest, sigma_scales_jacobian) {
  MockSimulator::reset();
  MockSimulator::ss_state.Gsc = 100.0;
  MockSimulator::basal_insulin = 1.0;

  auto data = make_id_data({0.0, 15.0, 30.0});
  std::vector<ParamDescriptor> descs = {{"kp1"}};
  auto base = Params::defaults_t1d();

  ObservationCost<MockSimulator> cost1(data, base, descs, 2.9);
  ObservationCost<MockSimulator> cost2(data, base, descs, 5.8);
  ASSERT_EQ(cost1.sigma_, 2.9);
  ASSERT_EQ(cost2.sigma_, 5.8);

  std::vector<double> theta = {0.1};
  const int n_obs = cost1.n_obs_;
  const int n_params = cost1.n_params_;

  // Jacobian for sigma=2.9
  std::vector<double> r1(n_obs), J1(n_obs * n_params);
  double* J1p[] = {J1.data()};
  const double* tp = theta.data();
  ASSERT_TRUE(cost1.Evaluate(&tp, r1.data(), J1p));

  // Jacobian for sigma=5.8
  std::vector<double> r2(n_obs), J2(n_obs * n_params);
  double* J2p[] = {J2.data()};
  ASSERT_TRUE(cost2.Evaluate(&tp, r2.data(), J2p));

  // J scales as 1/sigma: J2 = J1 * sigma1/sigma2 = J1 * 0.5
  double ratio = cost1.sigma_ / cost2.sigma_;  // = 0.5
  for (int i = 0; i < n_obs * n_params; ++i)
    ASSERT_TRUE(std::abs(J2[i] - J1[i] * ratio) < 1e-15);

  // Residuals also scale as 1/sigma
  for (int i = 0; i < n_obs; ++i)
    ASSERT_TRUE(std::abs(r2[i] - r1[i] * ratio) < 1e-15);

  // Scalar cost (SoftLOne): nonlinear in r, so not simply 2x or 0.5x
  double scalar1 = cost1.compute_scalar_cost(theta.data());
  double scalar2 = cost2.compute_scalar_cost(theta.data());
  EXPECT_TRUE(std::isfinite(scalar1));
  EXPECT_TRUE(std::isfinite(scalar2));
  // With sigma=2.9, |r| is larger → cost closer to linear (L1-like)
  // With sigma=5.8, |r| is smaller → cost closer to quadratic (L2-like)
  // Both should be positive and finite
  EXPECT_GT(scalar1, 0.0);
  EXPECT_GT(scalar2, 0.0);
  // Since sigma1 < sigma2, |r1| > |r2|, so cost1 > cost2
  EXPECT_GT(scalar1, scalar2);
}
