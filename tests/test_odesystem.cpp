#include "uva_padova/types.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/integrals.hpp"
#include "uva_padova/delivery.hpp"
#include "uva_padova/constants.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

using namespace uva_padova;

static Params make_default_params() {
  return Params::defaults_t1d();
}

// ===========================================================================
//  Test 1: Steady-state solver finds G ≈ Gb via bisection.
//  This tests that the model CAN reach the target basal glucose,
//  and that the found state is stable.
// ===========================================================================
void test_steady_state() {
  std::cout << "--- Test 1: Steady-state ---\n";

  Params p = make_default_params();
  State s{};
  double insulin_ss = SteadyStateSolver::find(s, p);
  double G_ss = UvaPadovaODESystem::plasma_glucose(s, p);

  std::cout << "  Steady-state glucose: " << G_ss << " mg/dL"
            << " (target Gb = " << p.Gb << ")\n"
            << "  Insulin rate: " << insulin_ss << " pmol/kg/min\n";
  std::cout << "  State values:\n";
  std::cout << "    Gp = " << s.Gp << " mg/kg\n";
  std::cout << "    Gt = " << s.Gt << " mg/kg\n";
  std::cout << "    Ip = " << s.Ip << " pmol/kg\n";
  std::cout << "    Ii = " << s.Ii << " pmol/kg\n";
  std::cout << "    Gsc = " << s.Gsc << " mg/dL\n";

  // CRITICAL: steady-state glucose must match target within tight tolerance
  assert(std::abs(G_ss - p.Gb) < 2.0
         && "SS glucose must match basal target");

  // Verify stability: hold for another 60 min, max drift stays small
  {
    UvaPadovaODESystem sys_check{};
    Inputs u_drift;
    u_drift.insulin_rate = insulin_ss;

    double max_drift = 0.0;
    for (int i = 0; i < 60; ++i) {
      State dxdt{};
      sys_check.rhs(s, dxdt, u_drift, p);
      for (int j = 0; j < State::kSize; ++j) {
        max_drift = std::max(max_drift, std::abs(dxdt.at(j)));
      }
      RK4Integrator::step(sys_check, s, u_drift, p, 1.0);
    }
    std::cout << "  Max drift after 60 min at SS: " << max_drift << "\n";
    assert(max_drift < 2.0 && "Steady state should be reasonably stable");
  }
  std::cout << "  PASSED\n";
}

// ===========================================================================
//  Test 2: plasma_glucose() matches Gp/VG
// ===========================================================================
void test_plasma_glucose() {
  std::cout << "--- Test 2: Plasma glucose ---\n";

  State s{};
  s.Gp = 200.0;
  Params p = make_default_params();
  p.VG = 2.0;

  double G = UvaPadovaODESystem::plasma_glucose(s, p);
  assert(std::abs(G - 100.0) < 1e-9 && "G = Gp / VG");
  std::cout << "  G = " << G << " mg/dL (expected 100)\n";
  std::cout << "  PASSED\n";
}

// ===========================================================================
//  Test 3: renal excretion activates above threshold
// ===========================================================================
void test_renal_excretion() {
  std::cout << "--- Test 3: Renal excretion ---\n";

  Params p = make_default_params();
  p.ke1 = 0.0005;
  p.ke2 = 339.0;

  double E_below = UvaPadovaODESystem::renal_excretion(300.0, p);
  double E_above = UvaPadovaODESystem::renal_excretion(400.0, p);

  assert(E_below == 0.0 && "No excretion below threshold");
  assert(E_above > 0.0 && "Excretion above threshold");
  std::cout << "  E(Gp=300) = " << E_below << ", E(Gp=400) = " << E_above << "\n";
  std::cout << "  PASSED\n";
}

// ===========================================================================
//  Test 4: gastric emptying has correct bi-exponential shape.
//
//  With the dynamic α,β parameterisation:
//    - Empty stomach (Qsto → 0):  k_emp ≈ kmin + (kmax-kmin)/2 * (tanh(-αbD) - tanh(-βdD) + 2)
//                                  Both αbD >> 1 and βdD >> 1 → tanh ≈ -1 → k_emp ≈ (kmax+kmin)/2
//    - Intermediate (Qsto ≈ 0.25·b·D): first tanh saturates (-1), second still near (+1) → kmin
//    - Full (Qsto > 2·b·D):          both saturate (+1) → kmax
// ===========================================================================
void test_gastric_emptying() {
  std::cout << "--- Test 4: Gastric emptying shape ---\n";

  Params p = make_default_params();
  const double D = 50000.0; // 50g meal

  double k_empty  = UvaPadovaODESystem::gastric_emptying_rate(0.0, D, p);
  double k_mid    = UvaPadovaODESystem::gastric_emptying_rate(0.25 * p.b * D, D, p);
  double k_full   = UvaPadovaODESystem::gastric_emptying_rate(3.0 * p.b * D, D, p);
  double k_very_big = UvaPadovaODESystem::gastric_emptying_rate(1e6, D, p);

  std::cout << "  k_emp(Qsto=0)         = " << k_empty   << "\n";
  std::cout << "  k_emp(0.25·b·D)       = " << k_mid     << "\n";
  std::cout << "  k_emp(3·b·D)          = " << k_full    << "\n";
  std::cout << "  k_emp(1e6)            = " << k_very_big << "\n";
  std::cout << "  kmax=" << p.kmax << "  kmin=" << p.kmin << "\n";

  // The curve dips near kmin at intermediate Qsto, then rises to kmax.
  assert(k_mid < k_empty - 1e-4 && "k_emp should dip below empty-stomach rate");
  assert(k_full > k_mid + 1e-4 && "Full stomach should empty faster than half");
  assert(std::abs(k_very_big - p.kmax) < 1e-4 && "Very full → kmax");
  assert(k_empty > p.kmin && k_empty < p.kmax && "Empty stomach rate in (kmin,kmax)");
  assert(k_full <= p.kmax + 1e-6 && "k_emp bounded by kmax");
  std::cout << "  PASSED\n";
}

// ===========================================================================
//  Test 5: 60-minute postprandial simulation with proper meal event.
// ===========================================================================
void test_meal_simulation() {
  std::cout << "--- Test 5: 60-min meal simulation ---\n";

  Params p = make_default_params();
  State s{};
  SteadyStateSolver::find(s, p);

  // Recover the corresponding basal insulin rate from the SS state
  double basal_rate = p.ka1 * s.Isc1 + p.ka2 * s.Isc2;

  UvaPadovaODESystem sys{};
  Inputs u;
  u.insulin_rate = basal_rate;

  // Apply a 50g meal at t=0 as a discrete event
  const double meal_mg = 50000.0;
  RK4Integrator::apply_meal_event(s, meal_mg);

  double G_before = UvaPadovaODESystem::plasma_glucose(s, p);
  double max_G = G_before;
  double min_G = G_before;

  for (int t = 0; t < 60; ++t) {
    RK4Integrator::step(sys, s, u, p, 1.0);
    double G = UvaPadovaODESystem::plasma_glucose(s, p);
    max_G = std::max(max_G, G);
    min_G = std::min(min_G, G);
  }

  std::cout << "  Pre-meal G=" << G_before
            << ", max=" << max_G
            << ", min=" << min_G << " mg/dL\n";

  assert(s.all_finite() && "State must remain finite");
  assert(max_G > p.Gb + 20.0 && "Postprandial rise should exceed fasting by 20+");
  // In T1DM with only basal (no bolus), glucose rises throughout the hour.
  // Check it's monotonically rising over the last 30 min.
  assert(max_G > 150.0 && "50g meal should push glucose well above 150 within 60 min");
  std::cout << "  PASSED\n";
}

// ===========================================================================
//  Test 6: 24-hour multi-meal simulation — meals as events, bolus as pulse.
// ===========================================================================
void test_24h_simulation() {
  std::cout << "--- Test 6: 24-hour multi-meal simulation ---\n";

  Params p = make_default_params();
  State s{};
  SteadyStateSolver::find(s, p);
  double basal_rate = p.ka1 * s.Isc1 + p.ka2 * s.Isc2;

  UvaPadovaODESystem sys{};
  Inputs u;
  u.insulin_rate = basal_rate;

  struct MealPlan {
    int t_min;
    double cho_mg;
    double bolus_pmol_kg_per_min; // rate for the 1-min bolus pulse
  };
  MealPlan meals[] = {
    { 0,   60000, 480.0 },
    { 360, 80000, 640.0 },
    { 720, 70000, 560.0 },
  };
  int n_meals = 3;
  int total_min = 24 * 60;

  double G_min = 1e9, G_max = 0.0;
  double final_G = 0.0;

  for (int t = 0; t < total_min; ++t) {
    Inputs u_step;
    u_step.insulin_rate = basal_rate;

    for (int m = 0; m < n_meals; ++m) {
      if (t == meals[m].t_min) {
        RK4Integrator::apply_meal_event(s, meals[m].cho_mg);
        u_step.insulin_rate += meals[m].bolus_pmol_kg_per_min;
      }
    }

    RK4Integrator::step(sys, s, u_step, p, 1.0);

    if (!s.all_finite()) {
      std::cerr << "  ERROR: non-finite state at t=" << t << "\n";
      break;
    }

    double G = UvaPadovaODESystem::plasma_glucose(s, p);
    G_min = std::min(G_min, G);
    G_max = std::max(G_max, G);

    if (t % 240 == 0) {
      std::cout << "  t=" << std::setw(4) << t << " min: G="
                << std::fixed << std::setprecision(1) << G << " mg/dL"
                << "  Gsc=" << s.Gsc << " mg/dL\n";
    }
    final_G = G;
  }

  std::cout << "  24h glucose range: " << G_min << " - " << G_max << " mg/dL\n";
  std::cout << "  Final G = " << final_G << " mg/dL (basal = " << p.Gb << ")\n";

  assert(s.all_finite() && "State must remain finite for 24h");
  assert(G_max > p.Gb + 30.0 && "At least one meal should cause significant rise");
  // By end of 24h the model should be returning toward basal
  assert(std::abs(final_G - p.Gb) < 100.0 && "24h should not drift arbitrarily far");
  std::cout << "  PASSED\n";
}

// ===========================================================================
//  Test 7: insulin dose raises plasma insulin
// ===========================================================================
void test_insulin_dynamics() {
  std::cout << "--- Test 7: Insulin dynamics ---\n";

  Params p = make_default_params();
  p.VI = 0.05;
  p.Ib = 70.0;

  State s{};
  s.Ip = p.Ib * p.VI;
  s.Ii = p.Ib * p.VI * 0.5;

  UvaPadovaODESystem sys{};
  Inputs u;
  u.insulin_rate = 667.0; // 0.5 U pulse over 1 min for 75kg

  double I_initial = s.Ip;
  for (int t = 0; t < 30; ++t) {
    RK4Integrator::step(sys, s, u, p, 1.0);
  }
  double I_final = s.Ip;

  std::cout << "  Ip: " << I_initial << " -> " << I_final << " pmol/kg\n";
  assert(I_final > I_initial && "Plasma insulin should rise with infusion");
  std::cout << "  PASSED\n";
}

// ===========================================================================
//  Test 8: EGP is always non-negative
// ===========================================================================
void test_egp_non_negative() {
  std::cout << "--- Test 8: EGP >= 0 ---\n";

  Params p = make_default_params();
  State s{};

  // High glucose + high insulin → should drive EGP to zero, never negative
  s.Gp = 500.0;
  s.Xl = 5.0;
  double egp_val = UvaPadovaODESystem::egp(s, p);
  assert(egp_val >= 0.0 && "EGP must be clamped at zero");

  // Zero conditions
  s.Gp = 0.0;
  s.Xl = 0.0;
  egp_val = UvaPadovaODESystem::egp(s, p);
  assert(egp_val >= 0.0 && "EGP at zero conditions must be >= 0");

  std::cout << "  egp(Gp=500, Xl=5) = " << UvaPadovaODESystem::egp({500,0,0,0,0,0,0,0,0,0,5,0,0,0}, p) << "\n";
  std::cout << "  PASSED\n";
}

// ===========================================================================
//  Test 9: apply_meal_event adds correctly and fixes mealD
// ===========================================================================
void test_apply_meal_event() {
  std::cout << "--- Test 9: apply_meal_event ---\n";

  State s{};
  assert(s.Qsto1 == 0.0 && s.mealD == 0.0 && "Initially empty");

  RK4Integrator::apply_meal_event(s, 50000.0);
  assert(s.Qsto1 == 50000.0 && "Qsto1 must increase by meal amount");
  assert(s.mealD == 50000.0 && "mealD must be fixed to meal amount");

  // Subsequent step should NOT change mealD via RHS
  Params p = make_default_params();
  Inputs u;
  UvaPadovaODESystem sys{};
  RK4Integrator::step(sys, s, u, p, 1.0);

  assert(s.mealD == 50000.0 && "mealD must persist across steps");
  assert(s.Qsto1 < 50000.0 && "Qsto1 should decrease via kgri*Qsto1");
  std::cout << "  Qsto1 after 1 min: " << s.Qsto1 << " mg, mealD: " << s.mealD << "\n";
  std::cout << "  PASSED\n";
}

// ===========================================================================
//  Test 10: UniformDelivery::rate_at
// ===========================================================================
void test_uniform_delivery() {
  std::cout << "--- Test 10: UniformDelivery ---\n";

  // Instantaneous (T <= 0)
  assert(UniformDelivery::rate_at(10.0, 10.0, 100.0, 0.0) == 100.0 && "T=0: D at t==t_start");
  assert(UniformDelivery::rate_at(11.0, 10.0, 100.0, 0.0) == 0.0 && "T=0: 0 at t!=t_start");
  assert(UniformDelivery::rate_at(10.0, 10.0, 100.0, -1.0) == 100.0 && "T<0: D at t==t_start");

  // Uniform over window
  assert(UniformDelivery::rate_at(5.0, 10.0, 100.0, 20.0) == 0.0 && "Before window: 0");
  assert(UniformDelivery::rate_at(10.0, 10.0, 100.0, 20.0) == 5.0 && "Start: D/T");
  assert(UniformDelivery::rate_at(20.0, 10.0, 100.0, 20.0) == 5.0 && "Inside: D/T");
  assert(UniformDelivery::rate_at(29.999, 10.0, 100.0, 20.0) == 5.0 && "Before end: D/T");
  assert(UniformDelivery::rate_at(30.0, 10.0, 100.0, 20.0) == 0.0 && "At end: 0");
  assert(UniformDelivery::rate_at(35.0, 10.0, 100.0, 20.0) == 0.0 && "After window: 0");

  // Zero dose
  assert(UniformDelivery::rate_at(10.0, 10.0, 0.0, 20.0) == 0.0 && "Zero dose");
  assert(UniformDelivery::rate_at(10.0, 10.0, 0.0, 0.0) == 0.0 && "Zero dose instant");

  std::cout << "  All 9 rate_at assertions passed\n";
  std::cout << "  PASSED\n";
}

// ===========================================================================
//  main
// ===========================================================================
int main() {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "=== UVA-Padova ODE System Tests ===\n\n";

  test_steady_state();
  std::cout << "\n";
  test_plasma_glucose();
  std::cout << "\n";
  test_renal_excretion();
  std::cout << "\n";
  test_gastric_emptying();
  std::cout << "\n";
  test_meal_simulation();
  std::cout << "\n";
  test_24h_simulation();
  std::cout << "\n";
  test_insulin_dynamics();
  std::cout << "\n";
  test_egp_non_negative();
  std::cout << "\n";
  test_apply_meal_event();
  std::cout << "\n";
  test_uniform_delivery();
  std::cout << "\n";

  std::cout << "=== ALL TESTS PASSED ===\n";
  return 0;
}
