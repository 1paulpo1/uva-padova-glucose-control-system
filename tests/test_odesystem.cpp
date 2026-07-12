#include <gtest/gtest.h>
#include <cmath>

#include "uva_padova/types.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/integrals.hpp"
#include "uva_padova/delivery.hpp"
#include "uva_padova/constants.hpp"

using namespace uva_padova;

namespace {

Params make_default_params() {
  return Params::defaults_t1d();
}

} // anonymous namespace

// ===========================================================================
//  Test 1: Steady-state solver finds G ≈ Gb via bisection.
// ===========================================================================
TEST(ODESystem, steady_state) {
  Params p = make_default_params();
  State s{};
  double insulin_ss = SteadyStateSolver::find(s, p);
  double G_ss = UvaPadovaODESystem::plasma_glucose(s, p);

  EXPECT_NEAR(G_ss, p.Gb, 2.0);

  // Verify stability: hold for another 60 min, max drift stays small
  {
    Inputs u_drift;
    u_drift.insulin_rate = insulin_ss;

    double max_drift = 0.0;
    for (int i = 0; i < 60; ++i) {
      State dxdt{};
      UvaPadovaODESystem::rhs(s, dxdt, u_drift, p);
      for (int j = 0; j < State::kSize; ++j) {
        max_drift = std::max(max_drift, std::abs(dxdt.at(j)));
      }
      RK4Integrator::step(UvaPadovaODESystem{}, s, u_drift, p, 1.0);
    }
    EXPECT_LT(max_drift, 2.0);
  }
}

// ===========================================================================
//  Test 2: plasma_glucose() matches Gp/VG
// ===========================================================================
TEST(ODESystem, plasma_glucose) {
  State s{};
  s.Gp = 200.0;
  Params p = make_default_params();
  p.VG = 2.0;

  double G = UvaPadovaODESystem::plasma_glucose(s, p);
  EXPECT_DOUBLE_EQ(G, 100.0);
}

// ===========================================================================
//  Test 3: renal excretion activates above threshold
// ===========================================================================
TEST(ODESystem, renal_excretion) {
  Params p = make_default_params();
  p.ke1 = 0.0005;
  p.ke2 = 339.0;

  double E_below = UvaPadovaODESystem::renal_excretion(300.0, p);
  double E_above = UvaPadovaODESystem::renal_excretion(400.0, p);

  EXPECT_EQ(E_below, 0.0);
  EXPECT_GT(E_above, 0.0);
}

// ===========================================================================
//  Test 4: gastric emptying has correct bi-exponential shape.
// ===========================================================================
TEST(ODESystem, gastric_emptying) {
  Params p = make_default_params();
  const double D = 50000.0;

  double k_empty  = UvaPadovaODESystem::gastric_emptying_rate(0.0, D, p);
  double k_mid    = UvaPadovaODESystem::gastric_emptying_rate(0.25 * p.b * D, D, p);
  double k_full   = UvaPadovaODESystem::gastric_emptying_rate(3.0 * p.b * D, D, p);
  double k_very_big = UvaPadovaODESystem::gastric_emptying_rate(1e6, D, p);

  EXPECT_LT(k_mid, k_empty - 1e-4);
  EXPECT_GT(k_full, k_mid + 1e-4);
  EXPECT_NEAR(k_very_big, p.kmax, 1e-4);
  EXPECT_GT(k_empty, p.kmin);
  EXPECT_LT(k_empty, p.kmax);
  EXPECT_LE(k_full, p.kmax + 1e-6);
}

// ===========================================================================
//  Test 5: 60-minute postprandial simulation with proper meal event.
// ===========================================================================
TEST(ODESystem, meal_simulation) {
  Params p = make_default_params();
  State s{};
  SteadyStateSolver::find(s, p);

  double basal_rate = p.ka1 * s.Isc1 + p.ka2 * s.Isc2;

  Inputs u;
  u.insulin_rate = basal_rate;

  const double meal_mg = 50000.0;
  RK4Integrator::apply_meal_event(s, meal_mg);

  double max_G = UvaPadovaODESystem::plasma_glucose(s, p);
  for (int t = 0; t < 60; ++t) {
    RK4Integrator::step(UvaPadovaODESystem{}, s, u, p, 1.0);
    double G = UvaPadovaODESystem::plasma_glucose(s, p);
    max_G = std::max(max_G, G);
  }

  EXPECT_TRUE(s.all_finite());
  EXPECT_GT(max_G, p.Gb + 20.0);
  EXPECT_GT(max_G, 150.0);
}

// ===========================================================================
//  Test 6: 24-hour multi-meal simulation.
// ===========================================================================
TEST(ODESystem, multi_meal_24h) {
  Params p = make_default_params();
  State s{};
  SteadyStateSolver::find(s, p);
  double basal_rate = p.ka1 * s.Isc1 + p.ka2 * s.Isc2;

  struct MealPlan { int t_min; double cho_mg; double bolus; };
  MealPlan meals[] = {
    { 0,   60000, 480.0 },
    { 360, 80000, 640.0 },
    { 720, 70000, 560.0 },
  };
  int n_meals = 3;
  int total_min = 24 * 60;

  double G_max = 0.0;
  double final_G = 0.0;

  for (int t = 0; t < total_min; ++t) {
    Inputs u_step;
    u_step.insulin_rate = basal_rate;

    for (int m = 0; m < n_meals; ++m) {
      if (t == meals[m].t_min) {
        RK4Integrator::apply_meal_event(s, meals[m].cho_mg);
        u_step.insulin_rate += meals[m].bolus;
      }
    }

    RK4Integrator::step(UvaPadovaODESystem{}, s, u_step, p, 1.0);

    ASSERT_TRUE(s.all_finite()) << "non-finite at t=" << t;

    double G = UvaPadovaODESystem::plasma_glucose(s, p);
    G_max = std::max(G_max, G);
    final_G = G;
  }

  EXPECT_GT(G_max, p.Gb + 30.0);
  EXPECT_NEAR(final_G, p.Gb, 100.0);
}

// ===========================================================================
//  Test 7: insulin dose raises plasma insulin
// ===========================================================================
TEST(ODESystem, insulin_dynamics) {
  Params p = make_default_params();
  p.VI = 0.05;
  p.Ib = 70.0;

  State s{};
  s.Ip = p.Ib * p.VI;
  s.Ii = p.Ib * p.VI * 0.5;

  Inputs u;
  u.insulin_rate = 667.0;

  double I_initial = s.Ip;
  for (int t = 0; t < 30; ++t) {
    RK4Integrator::step(UvaPadovaODESystem{}, s, u, p, 1.0);
  }
  EXPECT_GT(s.Ip, I_initial);
}

// ===========================================================================
//  Test 8: EGP is always non-negative
// ===========================================================================
TEST(ODESystem, egp_non_negative) {
  Params p = make_default_params();

  State s_high{};
  s_high.Gp = 500.0;
  s_high.Xl = 5.0;
  EXPECT_GE(UvaPadovaODESystem::egp(s_high, p), 0.0);

  State s_zero{};
  EXPECT_GE(UvaPadovaODESystem::egp(s_zero, p), 0.0);
}

// ===========================================================================
//  Test 9: apply_meal_event adds correctly and fixes mealD
// ===========================================================================
TEST(ODESystem, apply_meal_event) {
  State s{};
  EXPECT_EQ(s.Qsto1, 0.0);
  EXPECT_EQ(s.mealD, 0.0);

  RK4Integrator::apply_meal_event(s, 50000.0);
  EXPECT_EQ(s.Qsto1, 50000.0);
  EXPECT_EQ(s.mealD, 50000.0);

  Params p = make_default_params();
  Inputs u;
  RK4Integrator::step(UvaPadovaODESystem{}, s, u, p, 1.0);

  EXPECT_EQ(s.mealD, 50000.0);
  EXPECT_LT(s.Qsto1, 50000.0);
}

// ===========================================================================
//  Test 10: UniformDelivery::rate_at
// ===========================================================================
TEST(ODESystem, uniform_delivery) {
  EXPECT_EQ(UniformDelivery::rate_at(10.0, 10.0, 100.0, 0.0), 100.0);
  EXPECT_EQ(UniformDelivery::rate_at(11.0, 10.0, 100.0, 0.0), 0.0);
  EXPECT_EQ(UniformDelivery::rate_at(10.0, 10.0, 100.0, -1.0), 100.0);

  EXPECT_EQ(UniformDelivery::rate_at(5.0, 10.0, 100.0, 20.0), 0.0);
  EXPECT_EQ(UniformDelivery::rate_at(10.0, 10.0, 100.0, 20.0), 5.0);
  EXPECT_EQ(UniformDelivery::rate_at(20.0, 10.0, 100.0, 20.0), 5.0);
  EXPECT_EQ(UniformDelivery::rate_at(29.999, 10.0, 100.0, 20.0), 5.0);
  EXPECT_EQ(UniformDelivery::rate_at(30.0, 10.0, 100.0, 20.0), 0.0);
  EXPECT_EQ(UniformDelivery::rate_at(35.0, 10.0, 100.0, 20.0), 0.0);

  EXPECT_EQ(UniformDelivery::rate_at(10.0, 10.0, 0.0, 20.0), 0.0);
  EXPECT_EQ(UniformDelivery::rate_at(10.0, 10.0, 0.0, 0.0), 0.0);
}
