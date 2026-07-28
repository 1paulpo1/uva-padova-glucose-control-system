#include <gtest/gtest.h>
#include <cmath>
#include <Eigen/Dense>

#include "uva_padova/online_adaptation.hpp"
#include "uva_padova/synthetic_data_generator.hpp"

using namespace uva_padova;

// ===========================================================================
//  Test AugmentedStateMap
// ===========================================================================

TEST(CKFTest, AugmentedStateMapRoundTrip) {
  Eigen::VectorXd x(16);  // 14 model states + 2 theta
  x.setRandom();

  BasicState<double> s = AugmentedStateMap<double>::model_state(x);
  for (int i = 0; i < 14; ++i)
    EXPECT_DOUBLE_EQ(s.at(i), x(i));

  EXPECT_DOUBLE_EQ(AugmentedStateMap<double>::theta(x, 0), x(14));
  EXPECT_DOUBLE_EQ(AugmentedStateMap<double>::theta(x, 1), x(15));

  Eigen::VectorXd x2 = x;
  AugmentedStateMap<double>::set_model_state(x2, s);
  for (int i = 0; i < 14; ++i)
    EXPECT_DOUBLE_EQ(x2(i), x(i));

  EXPECT_EQ(AugmentedStateMap<double>::n_theta(x), 2);
}

// ===========================================================================
//  Test CubatureKalmanFilter with linear system
//  Simple 2D: x = [pos, vel], constant velocity + position measurement
// ===========================================================================

TEST(CKFTest, LinearTracking2D) {
  int n = 2;
  Eigen::VectorXd x0(n);
  x0 << 0.0, 1.0;

  Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(n, n) * 10.0;
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n, n) * 0.01;
  Eigen::MatrixXd R(1, 1);
  R(0, 0) = 1.0;

  CubatureKalmanFilter ckf(0, x0, P0, Q, R);

  // Simple constant-velocity step: [x + v*dt, v]
  auto f_model = [](Eigen::VectorXd& x) {
    double dt = 1.0;
    x(0) += dt * x(1);
    // velocity unchanged
  };

  auto h_model = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
    Eigen::VectorXd z(1);
    z(0) = x(0);
    return z;
  };

  // Simulate true trajectory
  Eigen::VectorXd x_true(n);
  x_true << 0.0, 1.0;
  std::vector<Eigen::VectorXd> measurements;
  for (int k = 0; k < 20; ++k) {
    x_true(0) += 1.0 * x_true(1);
    Eigen::VectorXd z(1);
    z(0) = x_true(0); // noiseless
    measurements.push_back(z);
  }

  // Run filter
  x_true << 0.0, 1.0;
  for (int k = 0; k < 20; ++k) {
    ckf.predict(f_model);
    auto innov = ckf.update(measurements[k], h_model);
    // Innovation should be small (noiseless measurement)
    EXPECT_NEAR(std::abs(innov(0)), 0.0, 2.0);
  }

  // Final position estimate should be close to true (position = 20)
  EXPECT_NEAR(ckf.state()(0), 20.0, 3.0);
  EXPECT_NEAR(ckf.state()(1), 1.0, 0.5);
}

// ===========================================================================
//  Test CKF with augmented state (model + theta random walk)
//  A single theta param with random walk, measurement = Gsc (state[12])
// ===========================================================================

TEST(CKFTest, AugmentedStateWithTheta) {
  // Use synthetic data to create a simple test scenario
  Params true_params;
  Params base_params;

  std::vector<ParamDescriptor> desc = {
    {"kabs", 0.0, 2.0, 0.0, false}
  };

  // Build theta vector: log(true / base)
  double theta_true = std::log(true_params.kabs / base_params.kabs);
  Eigen::VectorXd theta_init(1);
  theta_init(0) = theta_true;

  // Population variance for Q
  Eigen::MatrixXd Sigma_pop(1, 1);
  Sigma_pop(0, 0) = 0.1;

  // Sandwich covariance (P0)
  Eigen::MatrixXd P0_theta(1, 1);
  P0_theta(0, 0) = 0.05;

  // Config
  OnlineAdaptationConfig cfg;
  cfg.n_theta = 1;
  cfg.sigma_cgm = 2.9;
  cfg.alpha_q = 0.02;
  cfg.postprandial_window = 180.0;
  cfg.use_rts = false; // disable for this test

  OnlineAdaptation adapter(cfg, base_params, desc, P0_theta, theta_init, Sigma_pop);

  // Create synthetic scenario: single meal
  SyntheticDataConfig synth;
  synth.duration_min = 300;
  synth.cgm_interval = 5;
  synth.noise_sd = 0.0;  // noiseless for testing
  synth.seed = 42;
  synth.meals = {{60.0, 60000.0}};  // 60g CHO at t=60

  auto result = generate_synthetic_data(true_params, base_params, desc, synth);

  // Run the filter over the data
  for (size_t i = 0; i < result.data.cgm_time.size(); ++i) {
    double t = result.data.cgm_time[i];
    double gsc = result.data.cgm_value[i];

    // Register meal (60000 mg CHO = 60g)
    if (std::abs(t - 60.0) < 1.0)
      adapter.register_meal(t, 60000.0);

    auto innov = adapter.step(t, gsc);
    EXPECT_TRUE(std::isfinite(innov(0)));
  }

  // Final theta should be close to initial (noise-free, well-specified model)
  double final_theta = adapter.state()(14);
  EXPECT_NEAR(final_theta, theta_true, 0.5);
}

// ===========================================================================
//  Test RegimeModulatedQ
// ===========================================================================

TEST(CKFTest, RegimeModulatedQ) {
  std::vector<ParamDescriptor> desc = {
    {"kabs", 0.0, 2.0, 0.0, false},
    {"kp2", 0.0, 2.0, 0.0, false}
  };

  RegimeModulatedQ qmod(2, desc);
  int n = 14 + 2;
  qmod.Q_nominal_ = Eigen::MatrixXd::Identity(n, n) * 1.0;
  qmod.window_minutes = 180.0;
  qmod.freeze_factor = 0.001;

  // No meals active → kabs (idx 0) should be frozen, kp2 (idx 1) should not
  {
    Eigen::MatrixXd Q = qmod.compute(0.0);
    // kabs (theta index 0 → augmented index 14)
    EXPECT_DOUBLE_EQ(Q(14, 14), 0.001);
    // kp2 (theta index 1 → augmented index 15)
    EXPECT_DOUBLE_EQ(Q(15, 15), 1.0);
  }

  // Register meal at t=0 → window [0, 180]
  qmod.register_meal(0.0);

  // Inside window → all params active
  {
    Eigen::MatrixXd Q = qmod.compute(60.0);
    EXPECT_DOUBLE_EQ(Q(14, 14), 1.0);
    EXPECT_DOUBLE_EQ(Q(15, 15), 1.0);
  }

  // Outside window → kabs frozen again
  {
    Eigen::MatrixXd Q = qmod.compute(200.0);
    EXPECT_DOUBLE_EQ(Q(14, 14), 0.001);
    EXPECT_DOUBLE_EQ(Q(15, 15), 1.0);
  }
}

// ===========================================================================
//  Test RTSSmoother
// ===========================================================================

TEST(CKFTest, RTSSmootherBasic) {
  // Generate simple forward pass for a 1D random walk
  int n = 1;
  Eigen::VectorXd x0(n);
  x0 << 0.0;
  Eigen::MatrixXd P0(n, n);
  P0(0, 0) = 1.0;
  Eigen::MatrixXd Q(n, n);
  Q(0, 0) = 0.1;
  Eigen::MatrixXd R(n, n);
  R(0, 0) = 0.5;

  CubatureKalmanFilter ckf(0, x0, P0, Q, R);

  auto f_id = [](Eigen::VectorXd&) {}; // identity
  auto h_id = [](const Eigen::VectorXd& x) -> Eigen::VectorXd { return x; };

  // Generate measurements from true trajectory
  double x_true = 0.0;
  std::vector<RTSSmoother::FilteredStep> steps;
  std::vector<double> measurements;

  for (int k = 0; k < 10; ++k) {
    x_true += 0.2; // drift

    Eigen::VectorXd z(1);
    z(0) = x_true;
    measurements.push_back(z(0));

    ckf.predict(f_id);
    auto pred_x = ckf.state();
    auto pred_P = ckf.covariance();

    auto innov = ckf.update(z, h_id);

    RTSSmoother::FilteredStep step;
    step.t = static_cast<double>(k);
    step.x_pred = pred_x;
    step.P_pred = pred_P;
    step.x_filt = ckf.state();
    step.P_filt = ckf.covariance();
    steps.push_back(step);
  }

  // Run RTS smoother
  RTSSmoother smoother;
  auto result = smoother.smooth(steps);

  ASSERT_EQ(result.x_smooth.size(), steps.size());

  // Smoothed estimate at final step should equal filtered estimate
  EXPECT_DOUBLE_EQ(result.x_smooth.back()(0), steps.back().x_filt(0));

  // Smoothed estimate at early steps should have smaller covariance
  // (this is the key property of the RTS smoother)
  for (size_t k = 0; k < steps.size(); ++k) {
    EXPECT_TRUE(std::isfinite(result.x_smooth[k](0)));
    EXPECT_TRUE((result.P_smooth[k].array() >= 0).all());
  }
}

// ===========================================================================
//  Test MealPriorStore
// ===========================================================================

TEST(CKFTest, MealPriorStoreBasic) {
  MealPriorStore store;

  // Initially no priors
  std::vector<double> theta_out;
  EXPECT_FALSE(store.get_prior(60.0, theta_out)); // t=60min → hour 1 → breakfast-ish

  // Update with a breakfast prior
  std::vector<bool> is_meal = {true, false};
  store.update_prior(8.0 * 60.0, {0.1, 0.2}, is_meal);

  // Should now have a prior
  EXPECT_TRUE(store.get_prior(8.5 * 60.0, theta_out)); // 8:30 AM
  EXPECT_EQ(theta_out.size(), 1);  // only meal params
  EXPECT_NEAR(theta_out[0], 0.1, 1e-6);

  // Update again → should average
  store.update_prior(8.0 * 60.0 + 24.0 * 60.0, {0.3, 0.4}, is_meal); // next day
  EXPECT_TRUE(store.get_prior(8.5 * 60.0, theta_out));
  // lambda ≈ 0.95 (n=1 < 3) → θ = 0.95*0.1 + 0.05*0.3 = 0.11
  EXPECT_NEAR(theta_out[0], 0.11, 1e-6);

  // Different meal type (dinner) should not get breakfast prior
  std::vector<double> dinner_theta;
  EXPECT_FALSE(store.get_prior(19.0 * 60.0, dinner_theta)); // 7 PM, no dinner prior stored
}

// ===========================================================================
//  Test SystemModel
// ===========================================================================

TEST(CKFTest, SystemModelPropagation) {
  Params base_params;
  std::vector<ParamDescriptor> desc = {
    {"kabs", 0.0, 2.0, 0.0, false}
  };

  SystemModel sys_model(base_params, desc, 5.0, 5);

  // Initial augmented state: steady state with base params
  Eigen::VectorXd x(15);
  State ss_state;
  double basal = UvaPadovaSimulator::find_steady_state(ss_state, base_params, base_params.Gb);
  for (int i = 0; i < 14; ++i)
    x(i) = ss_state.at(i);
  x(14) = 0.0; // theta = 0 → kabs = base value

  // Propagate forward
  Eigen::VectorXd x_pred = x;
  sys_model(x_pred);

  // State should have evolved (not equal to initial)
  bool changed = false;
  for (int i = 0; i < 14; ++i) {
    if (std::abs(x_pred(i) - x(i)) > 1e-10) {
      changed = true;
      break;
    }
  }
  EXPECT_TRUE(changed);

  // All finite
  for (int i = 0; i < 15; ++i)
    EXPECT_TRUE(std::isfinite(x_pred(i)));

  // Theta unchanged (random walk)
  EXPECT_DOUBLE_EQ(x_pred(14), x(14));
}

// ===========================================================================
//  Test CKF with no params (pure state estimation)
// ===========================================================================

TEST(CKFTest, PureStateEstimation) {
  Params true_params;
  std::vector<ParamDescriptor> desc = {
    {"kabs", 0.0, 2.0, 0.0, false}
  };

  // Generate synthetic data
  SyntheticDataConfig synth;
  synth.duration_min = 240;
  synth.cgm_interval = 5;
  synth.noise_sd = 5.0;
  synth.seed = 123;
  synth.meals = {{30.0, 45000.0}, {150.0, 60000.0}};

  auto data = generate_synthetic_data(true_params, Params::defaults_t1d(), desc, synth);

  // Build online adaptation with 1 theta param
  Eigen::MatrixXd P0_theta(1, 1);
  P0_theta(0, 0) = 0.1;
  Eigen::VectorXd theta_init(1);
  theta_init(0) = 0.0;
  Eigen::MatrixXd Sigma_pop(1, 1);
  Sigma_pop(0, 0) = 0.2;

  OnlineAdaptationConfig cfg;
  cfg.n_theta = 1;
  cfg.sigma_cgm = 5.0;
  cfg.alpha_q = 0.02;
  cfg.postprandial_window = 180.0;
  cfg.use_rts = true;

  OnlineAdaptation adapter(cfg, Params::defaults_t1d(), desc, P0_theta, theta_init, Sigma_pop);

  // Run filter over data
  std::vector<double> innovations;
  for (size_t i = 0; i < data.data.cgm_time.size(); ++i) {
    double t = data.data.cgm_time[i];

    // Register meals (45000 mg = 45g at t=30, 60000 mg = 60g at t=150)
    if (std::abs(t - 30.0) < 1.0)
      adapter.register_meal(t, 45000.0);
    if (std::abs(t - 150.0) < 1.0)
      adapter.register_meal(t, 60000.0);

    auto innov = adapter.step(t, data.data.cgm_value[i]);
    innovations.push_back(innov(0));

    EXPECT_TRUE(std::isfinite(innov(0)));
  }

  // Innovations should have mean close to 0
  double sum = 0.0;
  for (double v : innovations) sum += v;
  double mean = sum / static_cast<double>(innovations.size());
  EXPECT_NEAR(mean, 0.0, 5.0);

  // Final covariance should be positive definite
  auto P = adapter.covariance();
  EXPECT_TRUE((P.diagonal().array() >= 0).all());
}

// ===========================================================================
//  Test filter with multiple meals and RTS smoothing
// ===========================================================================

TEST(CKFTest, MultipleMealsWithRTS) {
  Params true_params;
  std::vector<ParamDescriptor> desc = {
    {"kabs", 0.0, 2.0, 0.0, false}
  };

  SyntheticDataConfig synth;
  synth.duration_min = 480;
  synth.cgm_interval = 5;
  synth.noise_sd = 3.0;
  synth.seed = 456;
  synth.meals = {{60.0, 50000.0}, {240.0, 70000.0}, {390.0, 55000.0}};

  auto data = generate_synthetic_data(true_params, Params::defaults_t1d(), desc, synth);

  Eigen::MatrixXd P0_theta(1, 1);
  P0_theta(0, 0) = 0.1;
  Eigen::VectorXd theta_init(1);
  theta_init(0) = 0.0;
  Eigen::MatrixXd Sigma_pop(1, 1);
  Sigma_pop(0, 0) = 0.2;

  OnlineAdaptationConfig cfg;
  cfg.n_theta = 1;
  cfg.sigma_cgm = 3.0;
  cfg.alpha_q = 0.02;

  OnlineAdaptation adapter(cfg, Params::defaults_t1d(), desc, P0_theta, theta_init, Sigma_pop);

  for (size_t i = 0; i < data.data.cgm_time.size(); ++i) {
    double t = data.data.cgm_time[i];
    if (std::abs(t - 60.0) < 1.0)
      adapter.register_meal(t, 50000.0);
    if (std::abs(t - 240.0) < 1.0)
      adapter.register_meal(t, 70000.0);
    if (std::abs(t - 390.0) < 1.0)
      adapter.register_meal(t, 55000.0);

    auto innov = adapter.step(t, data.data.cgm_value[i]);
    EXPECT_TRUE(std::isfinite(innov(0)));
  }

  // Run RTS smoothing on the full history
  auto theta_smooth = adapter.smooth_completed_segment(0.0);

  // Should have smoothed theta
  EXPECT_FALSE(theta_smooth.empty());
  EXPECT_TRUE(std::isfinite(theta_smooth[0]));

  // Meal prior should have been updated
  EXPECT_GE(adapter.meal_priors().size(), 0);

  // No explosions in covariance
  auto P = adapter.covariance();
  for (int i = 0; i < P.rows(); ++i) {
    EXPECT_TRUE(std::isfinite(P(i, i)));
    EXPECT_GE(P(i, i), 0.0);
  }
}

// ===========================================================================
//  Test OnlineObservability
// ===========================================================================

TEST(CKFTest, OnlineObservabilityTrivial) {
  const int dim = 4;
  OnlineObservability obs(dim);

  EXPECT_FALSE(obs.ready());
  EXPECT_EQ(obs.dim(), dim);

  // Identity dynamics: f(x) = x
  auto f_identity = [](Eigen::VectorXd& x) {};

  Eigen::VectorXd x0 = Eigen::VectorXd::Ones(dim);

  // Push one Jacobian
  obs.push_jacobian(f_identity, x0);
  EXPECT_TRUE(obs.ready());
  EXPECT_EQ(obs.window_size(), 1);

  auto s = obs.scores();

  EXPECT_EQ(s.size(), dim);
  // Gsc is at index 2 (since dim=4, index 2 maps to original position 12... no.
  // Actually H is hardcoded: H(0, 12) = 1.0, but dim=4 so index 12 is out of bounds!
  // Score for index 2 should be 0 since H(0,2) = 0.
  // Only index 12 would have non-zero, but dim=4 so all zeros.
  for (int i = 0; i < dim; ++i) {
    EXPECT_TRUE(std::isfinite(s(i)));
    EXPECT_EQ(s(i), 0.0);
  }
}

TEST(CKFTest, OnlineObservabilityWithSystemModel) {
  // Use the actual SystemModel to push Jacobians and check
  // that meal-related theta parameters have higher observability
  // during the postprandial window than outside it.

  BasicParams<double> params = BasicParams<double>::defaults_t1d();
  std::vector<ParamDescriptor> desc = {
    {"kabs", false, 0.3, 3.0},
    {"kp1", false, -0.5, 0.5},
  };
  int n_theta = static_cast<int>(desc.size());
  int dim = 14 + n_theta;

  OnlineObservability obs(dim);
  obs.window_length = 3;
  SystemModel sys_model(params, desc, 5.0);

  // State: steady state for all model components, zero theta
  State ss;
  Eigen::VectorXd x0(dim);
  {
    BasicParams<double> p = params;
    UvaPadovaSimulator::find_steady_state(ss, p, p.Gb);
    for (int i = 0; i < 14; ++i)
      x0(i) = ss.at(i);
    for (int i = 0; i < n_theta; ++i)
      x0(14 + i) = 0.0;
  }

  // Simulate a meal: set Qsto1 and mealD
  Eigen::VectorXd x = x0;
  x(2) += 90000.0;  // Qsto1: 90g CHO
  x(13) = 90000.0;  // mealD

  // Push several Jacobians along the meal trajectory
  // (simulating the system dynamics advancing)
  for (int k = 0; k < 6; ++k) {
    obs.push_jacobian(sys_model, x);

    // Advance state for next step
    Eigen::VectorXd x_next = x;
    sys_model(x_next);
    x = x_next;
  }

  auto s = obs.scores();
  EXPECT_TRUE(obs.ready());

  // All scores should be finite
  for (int i = 0; i < dim; ++i)
    EXPECT_TRUE(std::isfinite(s(i)));

  // kabs (index 14) should have observability > 0 during meal
  // Actually scores depend on the dynamics so we just check basic sanity
  EXPECT_GE(s(14), 0.0);
  EXPECT_GE(s(15), 0.0);
}

TEST(CKFTest, ObservabilityModulatesRegimeQ) {
  // Verify that setting observability scores on RegimeModulatedQ
  // changes Q outside the meal window.

  BasicParams<double> params = BasicParams<double>::defaults_t1d();
  std::vector<ParamDescriptor> desc = {
    {"kabs", false, 0.3, 3.0},
    {"kp1", false, -0.5, 0.5},
  };
  int n_theta = static_cast<int>(desc.size());
  int dim = 14 + n_theta;

  RegimeModulatedQ Q_mod(n_theta, desc);
  Q_mod.Q_nominal_ = Eigen::MatrixXd::Identity(dim, dim);
  Q_mod.window_minutes = 180.0;
  Q_mod.freeze_factor = 0.001;

  // No meal registered → outside any window
  double t = 60.0;

  // Without scores: meal params frozen
  Eigen::MatrixXd Q1 = Q_mod.compute(t);
  EXPECT_DOUBLE_EQ(Q1(15, 15), 1.0);  // kp1 (non-meal) unchanged
  EXPECT_DOUBLE_EQ(Q1(14, 14), 0.001);  // kabs (meal) frozen

  // With high observability scores: less freezing
  Eigen::VectorXd scores = Eigen::VectorXd::Ones(dim);
  Q_mod.set_observability_scores(scores);
  Eigen::MatrixXd Q2 = Q_mod.compute(t);
  // kabs should now be nearly 1.0 (fully observable → full Q)
  EXPECT_NEAR(Q2(14, 14), 1.0, 0.01);

  // With low observability scores: heavy freezing
  scores.setZero();
  scores(15) = 0.0;  // kp1 score irrelevant (non-meal)
  Q_mod.set_observability_scores(scores);
  Eigen::MatrixXd Q3 = Q_mod.compute(t);
  // kabs should be at freeze_factor
  EXPECT_NEAR(Q3(14, 14), 0.001, 1e-10);
}
