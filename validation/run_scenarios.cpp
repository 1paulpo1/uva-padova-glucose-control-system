#include "uva_padova/types.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/integrals.hpp"
#include "uva_padova/delivery.hpp"
#include "uva_padova/constants.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <utility>

using namespace uva_padova;

// ===========================================================================
//  CSV helpers
// ===========================================================================
static void write_csv(const std::string& path, const std::vector<std::string>& header,
                      const std::vector<std::vector<double>>& rows) {
  std::ofstream f(path);
  if (!f.is_open()) {
    std::cerr << "ERROR: cannot write " << path << "\n";
    return;
  }
  for (size_t i = 0; i < header.size(); ++i) {
    f << header[i];
    if (i + 1 < header.size()) f << ",";
  }
  f << "\n";
  for (auto& row : rows) {
    for (size_t i = 0; i < row.size(); ++i) {
      f << row[i];
      if (i + 1 < row.size()) f << ",";
    }
    f << "\n";
  }
  std::cout << "  Wrote " << rows.size() << " rows -> " << path << "\n";
}

static void write_csv_rows(const std::string& path, const std::vector<std::string>& header,
                           const std::vector<std::vector<double>>& rows) {
  write_csv(path, header, rows);
}

// ===========================================================================
//  Fast-forward to steady state and record hold stability
// ===========================================================================
static void scenario_ss_hold(const std::string& output_dir) {
  std::cout << "\n=== Scenario: Steady-state hold (360 min) ===\n";
  Params p = Params::defaults_t1d();
  State s{};
  double basal_rate = SteadyStateSolver::find(s, p);

  UvaPadovaODESystem sys{};
  Inputs u;
  u.insulin_rate = basal_rate;

  std::vector<std::string> header = {"t_min","G","Gsc","Gp","Gt","Ip","Ii","Isc1","Isc2","I_prime","Xl","X","Qsto1","Qsto2","Qgut"};
  std::vector<std::vector<double>> rows;
  for (int t = 0; t <= 360; ++t) {
    double G = UvaPadovaODESystem::plasma_glucose(s, p);
    rows.push_back({
      (double)t, G, s.Gsc, s.Gp, s.Gt, s.Ip, s.Ii, s.Isc1, s.Isc2,
      s.I_prime, s.Xl, s.X, s.Qsto1, s.Qsto2, s.Qgut
    });
    RK4Integrator::step(sys, s, u, p, 1.0);
  }
  write_csv_rows(output_dir + "/ss_hold.csv", header, rows);
}

// ===========================================================================
//  Gastric emptying rate curve for multiple meal sizes
// ===========================================================================
static void scenario_gastric_emptying(const std::string& output_dir) {
  std::cout << "\n=== Scenario: Gastric emptying curve ===\n";
  Params p = Params::defaults_t1d();

  std::vector<double> meal_sizes = {0.0, 30000.0, 50000.0, 80000.0};
  std::vector<std::string> header = {"Qsto","D","k_emp"};
  std::vector<std::vector<double>> rows;

  for (double D : meal_sizes) {
    for (double Qsto = 0.0; Qsto <= 300000.0; Qsto += 500.0) {
      double k = UvaPadovaODESystem::gastric_emptying_rate(Qsto, D, p);
      rows.push_back({Qsto, D, k});
    }
  }
  write_csv_rows(output_dir + "/gastric_emptying.csv", header, rows);
}

// ===========================================================================
//  60-min meal simulation (50g CHO, basal only, no bolus)
// ===========================================================================
static void scenario_meal_60min(const std::string& output_dir) {
  std::cout << "\n=== Scenario: 60-min postprandial (50g, no bolus) ===\n";
  Params p = Params::defaults_t1d();
  State s{};
  double basal_rate = SteadyStateSolver::find(s, p);

  UvaPadovaODESystem sys{};
  Inputs u;
  u.insulin_rate = basal_rate;

  RK4Integrator::apply_meal_event(s, 50000.0);

  std::vector<std::string> header = {"t_min","G","Gsc","Gp","Gt","Ip","Ii","Isc1","Isc2","Qsto1","Qsto2","Qgut","Ra_meal","EGP","Uid"};
  std::vector<std::vector<double>> rows;
  for (int t = 0; t <= 60; ++t) {
    double G = UvaPadovaODESystem::plasma_glucose(s, p);
    State dxdt{};
    sys.rhs(s, dxdt, u, p);
    rows.push_back({
      (double)t, G, s.Gsc, s.Gp, s.Gt, s.Ip, s.Ii, s.Isc1, s.Isc2,
      s.Qsto1, s.Qsto2, s.Qgut,
      p.f * p.kabs * s.Qgut / p.BW,
      UvaPadovaODESystem::egp(s, p),
      UvaPadovaODESystem::utilization_insulin_dependent(s, p)
    });
    RK4Integrator::step(sys, s, u, p, 1.0);
  }
  write_csv_rows(output_dir + "/meal_60min.csv", header, rows);
}

// ===========================================================================
//  24-hour multi-meal simulation
// ===========================================================================
static void scenario_24h(const std::string& output_dir) {
  std::cout << "\n=== Scenario: 24-hour multi-meal ===\n";
  Params p = Params::defaults_t1d();
  State s{};
  double basal_rate = SteadyStateSolver::find(s, p);

  UvaPadovaODESystem sys{};
  Inputs u;
  u.insulin_rate = basal_rate;

  struct Meal { int t_min; double cho_mg; double bolus_rate; };
  Meal meals[] = {
    { 0,   60000, 480.0 },
    { 360, 80000, 640.0 },
    { 720, 70000, 560.0 },
  };

  std::vector<std::string> header = {"t_min","G","Gsc","Ip","Ii","Isc1","Isc2","Qsto1","Qsto2","Qgut","meal_flag"};
  std::vector<std::vector<double>> rows;
  for (int t = 0; t < 1440; ++t) {
    Inputs u_step;
    u_step.insulin_rate = basal_rate;

    double meal_flag = 0.0;
    for (auto& m : meals) {
      if (t == m.t_min) {
        RK4Integrator::apply_meal_event(s, m.cho_mg);
        u_step.insulin_rate += m.bolus_rate;
        meal_flag = m.cho_mg;
      }
    }

    double G = UvaPadovaODESystem::plasma_glucose(s, p);
    rows.push_back({
      (double)t, G, s.Gsc, s.Ip, s.Ii, s.Isc1, s.Isc2,
      s.Qsto1, s.Qsto2, s.Qgut, meal_flag
    });

    RK4Integrator::step(sys, s, u_step, p, 1.0);
  }
  write_csv_rows(output_dir + "/day_24h.csv", header, rows);
}

// ===========================================================================
//  Insulin pulse response (180 min after a large bolus)
// ===========================================================================
static void scenario_insulin_pulse(const std::string& output_dir) {
  std::cout << "\n=== Scenario: Insulin pulse response (180 min) ===\n";
  Params p = Params::defaults_t1d();
  State s{};
  double basal_rate = SteadyStateSolver::find(s, p);

  UvaPadovaODESystem sys{};

  std::vector<std::string> header = {"t_min","G","Ip","Ii","Isc1","Isc2","X","I_prime","Xl"};
  std::vector<std::vector<double>> rows;

  // Baseline
  Inputs u_base;
  u_base.insulin_rate = basal_rate;

  for (int t = -30; t <= 180; ++t) {
    Inputs u_step = u_base;
    if (t == 0) {
      u_step.insulin_rate += 0.5 * constants::kInsulinUnits_To_Pmol / p.BW; // 0.5 U pulse
    }

    if (t >= 0) {
      double G = UvaPadovaODESystem::plasma_glucose(s, p);
      rows.push_back({
        (double)t, G, s.Ip, s.Ii, s.Isc1, s.Isc2, s.X, s.I_prime, s.Xl
      });
    }

    RK4Integrator::step(sys, s, u_step, p, 1.0);
  }
  write_csv_rows(output_dir + "/insulin_pulse.csv", header, rows);
}

// ===========================================================================
//  Comparison scenario: pulse (T=1) vs uniform (T=15/30) for meal & insulin
// ===========================================================================
static void scenario_comparison_delivery(const std::string& output_dir) {
  std::cout << "\n=== Scenario: Delivery comparison (pulse vs uniform) ===\n";

  Params p = Params::defaults_t1d();
  State s_pulse{}, s_uniform{};
  double basal_rate = SteadyStateSolver::find(s_pulse, p);
  s_uniform = s_pulse;
  const double meal_cho = 50000.0;
  const double bolus_dose = 0.5 * constants::kInsulinUnits_To_Pmol / p.BW; // 0.5 U, pmol/kg/min for 1 min

  UvaPadovaODESystem sys{};

  // --- Meal comparison: pulse (T=1) vs uniform (T=15) ---
  // Simulate 120 min, basal insulin constant, meal at t=0
  Inputs u_base;
  u_base.insulin_rate = basal_rate;

  std::vector<std::string> meal_header = {
    "t_min","G_pulse","Gsc_pulse","Qsto1_pulse","G_uniform","Gsc_uniform","Qsto1_uniform"
  };
  std::vector<std::vector<double>> meal_rows;

  // Reset both states from SS, apply meal with different T
  State s_meal_pulse = s_pulse;
  State s_meal_uniform = s_uniform;

  for (int t = 0; t <= 120; ++t) {
    // Pulse branch
    s_meal_pulse.Qsto1 += UniformDelivery::rate_at(double(t), 0.0, meal_cho, 1.0);
    if (t == 0) s_meal_pulse.mealD = meal_cho;

    // Uniform branch
    s_meal_uniform.Qsto1 += UniformDelivery::rate_at(double(t), 0.0, meal_cho, 15.0);
    if (t == 0) s_meal_uniform.mealD = meal_cho;

    double Gp = UvaPadovaODESystem::plasma_glucose(s_meal_pulse, p);
    double Gu = UvaPadovaODESystem::plasma_glucose(s_meal_uniform, p);
    meal_rows.push_back({
      double(t), Gp, s_meal_pulse.Gsc, s_meal_pulse.Qsto1,
      Gu, s_meal_uniform.Gsc, s_meal_uniform.Qsto1
    });

    RK4Integrator::step(sys, s_meal_pulse, u_base, p, 1.0);
    RK4Integrator::step(sys, s_meal_uniform, u_base, p, 1.0);
  }
  write_csv_rows(output_dir + "/comparison_meal.csv", meal_header, meal_rows);

  // --- Insulin comparison: pulse (T=1) vs uniform (T=30) ---
  // Simulate 240 min, steady-state, bolus at t=0
  State s_ins_pulse = s_pulse;
  State s_ins_uniform = s_uniform;

  std::vector<std::string> ins_header = {
    "t_min","Ip_pulse","Ii_pulse","X_pulse","Isc1_pulse","Isc2_pulse",
    "Ip_uniform","Ii_uniform","X_uniform","Isc1_uniform","Isc2_uniform"
  };
  std::vector<std::vector<double>> ins_rows;

  for (int t = 0; t <= 240; ++t) {
    Inputs u_pulse, u_uniform;
    u_pulse.insulin_rate = basal_rate + UniformDelivery::rate_at(double(t), 0.0, bolus_dose, 1.0);
    u_uniform.insulin_rate = basal_rate + UniformDelivery::rate_at(double(t), 0.0, bolus_dose, 30.0);

    ins_rows.push_back({
      double(t),
      s_ins_pulse.Ip, s_ins_pulse.Ii, s_ins_pulse.X, s_ins_pulse.Isc1, s_ins_pulse.Isc2,
      s_ins_uniform.Ip, s_ins_uniform.Ii, s_ins_uniform.X, s_ins_uniform.Isc1, s_ins_uniform.Isc2
    });

    RK4Integrator::step(sys, s_ins_pulse, u_pulse, p, 1.0);
    RK4Integrator::step(sys, s_ins_uniform, u_uniform, p, 1.0);
  }
  write_csv_rows(output_dir + "/comparison_insulin.csv", ins_header, ins_rows);
}

// ===========================================================================
//  main
// ===========================================================================
int main(int argc, char** argv) {
  std::string output_dir = VALIDATION_OUTPUT_DIR;
  if (argc > 1) {
    output_dir = argv[1];
  }
  std::cout << "Validation output directory: " << output_dir << "\n";

  scenario_ss_hold(output_dir);
  scenario_gastric_emptying(output_dir);
  scenario_meal_60min(output_dir);
  scenario_24h(output_dir);
  scenario_insulin_pulse(output_dir);
  scenario_comparison_delivery(output_dir);

  std::cout << "\n=== All scenarios complete ===\n";
  return 0;
}
