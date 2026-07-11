// =============================================================================
// Test the identification pipeline on real patient data.
//
// Usage:
//     ./test_real_data <cgm_csv> <basal_csv> [n_restarts] [cmaes_evals] [mode]
//     mode: 'config' (default, JSON config — 21 active params, greedy-5 fixed)
//           '3'     (kp2+Vm0+p2U, legacy), '6' (+ka1+ka2+kd, legacy),
//           '9'     (+kabs+Gb+VG, legacy)
//
// Defaults: 3 restarts, 500 evals each.
// =============================================================================

#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <algorithm>

#include <Eigen/Dense>

#include "uva_padova/types.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/integrals.hpp"
#include "uva_padova/delivery.hpp"
#include "uva_padova/cost_function.hpp"
#include "uva_padova/optimizer.hpp"
#include "uva_padova/diagnostics.hpp"
#include "uva_padova/constants.hpp"
#include "uva_padova/config_loader.hpp"

#include "csv_reader.hpp"

using namespace uva_padova;

int main(int argc, char* argv[]) {
  std::cout << std::scientific << std::setprecision(6);

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <cgm_csv> <basal_csv> [n_restarts] [cmaes_evals] [mode]\n";
    std::cerr << "  mode: 'config' (default, JSON — 21 active, greedy-5 fixed),\n";
    std::cerr << "        '3' (kp2+Vm0+p2U), '6' (+ka1+ka2+kd), '9' (+kabs+Gb+VG)\n";
    return 1;
  }

  std::string cgm_csv = argv[1];
  std::string basal_csv = argv[2];
  int n_restarts = (argc > 3) ? std::atoi(argv[3]) : 3;
  int cmaes_evals = (argc > 4) ? std::atoi(argv[4]) : 500;
  std::string mode = (argc > 5) ? argv[5] : "config";

  // ---- Choose params to identify ----
  std::vector<ParamDescriptor> desc;
  bool use_config = (mode == "config");
  if (!use_config) {
    if (mode == "6") {
    desc = {
      { "kp2",  0.0, 1.0, 0.0 },
      { "Vm0",  0.0, 1.0, 0.0 },
      { "p2U",  0.0, 1.0, 0.0 },
      { "ka1",  0.0, 1.0, 0.0 },
      { "ka2",  0.0, 1.0, 0.0 },
      { "kd",   0.0, 1.0, 0.0 },
    };
  } else if (mode == "9") {
    desc = {
      { "kp2",  0.0, 1.0, 0.0 },
      { "Vm0",  0.0, 1.0, 0.0 },
      { "p2U",  0.0, 1.0, 0.0 },
      { "ka1",  0.0, 1.0, 0.0 },
      { "ka2",  0.0, 1.0, 0.0 },
      { "kd",   0.0, 1.0, 0.0 },
      { "kabs", 0.0, 1.0, 0.0 },
      { "Gb",   0.0, 0.2, 0.0 },  // tight prior — Gb ≈ CGM median
      { "VG",   0.0, 1.0, 0.0 },
    };
    } else {
      desc = {
        { "kp2",  0.0, 1.0, 0.0 },
        { "Vm0",  0.0, 1.0, 0.0 },
        { "p2U",  0.0, 1.0, 0.0 },
      };
    }
  }
  std::cout << "  Mode: ";
  if (use_config) {
    std::cout << "config (JSON)";
  } else {
    std::cout << desc.size() << " params (";
    for (size_t i = 0; i < desc.size(); ++i)
      std::cout << (i ? ", " : "") << desc[i].name;
    std::cout << ")";
  }
  std::cout << "\n";

  Params base_params = Params::defaults_t1d();

  // ---- Load data (first pass) to estimate Gb from CGM ----
  auto data = RealDataReader::load_from_csv(cgm_csv, basal_csv, base_params);
  if (data.cgm_time.empty()) {
    std::cerr << "Failed to load data\n";
    return 1;
  }

  // Estimate Gb from CGM: use median of lower half (Q2 = median)
  {
    auto sorted = data.cgm_value;
    std::sort(sorted.begin(), sorted.end());
    double Gb_est = sorted[sorted.size() / 2];  // median
    std::cout << "  Estimated Gb = " << Gb_est << " mg/dL (median of "
              << sorted.size() << " CGM values)\n";
    base_params.Gb = Gb_est;
    // Reload with adjusted Gb so steady-state and basal encoding are consistent
    data = RealDataReader::load_from_csv(cgm_csv, basal_csv, base_params);
  }

  // ---- Run pipeline ----
  std::cout << "\n=== Identification pipeline ===\n";

  std::vector<ParamDescriptor> full_desc;
  auto result = use_config
    ? IdentificationPipeline::run_from_config(data, base_params,
        UVA_PADOVA_CONFIG_DIR "/default_params.json",
        n_restarts, cmaes_evals, &full_desc)
    : IdentificationPipeline::run(data, base_params, desc, n_restarts, cmaes_evals);

  if (!result.success) {
    std::cerr << "Pipeline failed\n";
    return 1;
  }

  const auto& active_desc = use_config ? active_descriptors(full_desc) : desc;

  // ---- Print result ----
  std::cout << "\n=== Pipeline result ===\n";
  std::cout << "  cost=" << result.cost << "\n";
  std::cout << "  success=" << (result.success ? "yes" : "no") << "\n";
  std::cout << "  theta = [";
  for (size_t i = 0; i < result.params.size(); ++i)
    std::cout << (i ? ", " : "") << result.params[i];
  std::cout << "]\n";

  auto recovered = result.to_params(base_params, active_desc);
  auto default_p = Params::defaults_t1d();

  for (const auto& d : active_desc) {
    double rec_val = 0.0, def_val = 0.0;
    if (d.name == "VG")   { rec_val = recovered.VG;   def_val = default_p.VG; }
    if (d.name == "k1")   { rec_val = recovered.k1;   def_val = default_p.k1; }
    if (d.name == "k2")   { rec_val = recovered.k2;   def_val = default_p.k2; }
    if (d.name == "m1")   { rec_val = recovered.m1;   def_val = default_p.m1; }
    if (d.name == "m2")   { rec_val = recovered.m2;   def_val = default_p.m2; }
    if (d.name == "m4")   { rec_val = recovered.m4;   def_val = default_p.m4; }
    if (d.name == "kmax") { rec_val = recovered.kmax; def_val = default_p.kmax; }
    if (d.name == "kmin") { rec_val = recovered.kmin; def_val = default_p.kmin; }
    if (d.name == "kabs") { rec_val = recovered.kabs; def_val = default_p.kabs; }
    if (d.name == "kgri") { rec_val = recovered.kgri; def_val = default_p.kgri; }
    if (d.name == "f")    { rec_val = recovered.f;    def_val = default_p.f; }
    if (d.name == "b")    { rec_val = recovered.b;    def_val = default_p.b; }
    if (d.name == "d")    { rec_val = recovered.d;    def_val = default_p.d; }
    if (d.name == "kp1")  { rec_val = recovered.kp1;  def_val = default_p.kp1; }
    if (d.name == "kp2")  { rec_val = recovered.kp2;  def_val = default_p.kp2; }
    if (d.name == "kp3")  { rec_val = recovered.kp3;  def_val = default_p.kp3; }
    if (d.name == "ki")   { rec_val = recovered.ki;   def_val = default_p.ki; }
    if (d.name == "Vm0")  { rec_val = recovered.Vm0;  def_val = default_p.Vm0; }
    if (d.name == "Vmx")  { rec_val = recovered.Vmx;  def_val = default_p.Vmx; }
    if (d.name == "p2U")  { rec_val = recovered.p2U;  def_val = default_p.p2U; }
    if (d.name == "r1")   { rec_val = recovered.r1;   def_val = default_p.r1; }
    if (d.name == "Gth")  { rec_val = recovered.Gth;  def_val = default_p.Gth; }
    if (d.name == "ke1")  { rec_val = recovered.ke1;  def_val = default_p.ke1; }
    if (d.name == "ke2")  { rec_val = recovered.ke2;  def_val = default_p.ke2; }
    if (d.name == "ka1")  { rec_val = recovered.ka1;  def_val = default_p.ka1; }
    if (d.name == "ka2")  { rec_val = recovered.ka2;  def_val = default_p.ka2; }
    if (d.name == "kd")   { rec_val = recovered.kd;   def_val = default_p.kd; }
    if (d.name == "Ts")   { rec_val = recovered.Ts;   def_val = default_p.Ts; }
    if (d.name == "Gb")   { rec_val = recovered.Gb;   def_val = default_p.Gb; }
    std::cout << "  Recovered " << d.name << " = " << rec_val
              << " (default=" << def_val << ")\n";
  }

  // ---- Posterior cov ----
  std::cout << "\n=== Posterior cov (from pipeline) ===\n";
  if (result.cov.rows() > 0) {
    auto print_corr = [&](int i, int j) {
      double c = result.cov(i,j) / std::sqrt(result.cov(i,i) * result.cov(j,j));
      std::cout << "  corr(" << active_desc[i].name << "," << active_desc[j].name << ")=" << c << "\n";
    };
    for (size_t i = 0; i < active_desc.size(); ++i)
      for (size_t j = i+1; j < active_desc.size(); ++j)
        print_corr(static_cast<int>(i), static_cast<int>(j));
  } else {
    std::cout << "  (empty)\n";
  }

  // ---- Gate (a) ----
  {
    auto gate_a = diagnose_multiplicity(result.all_restarts, active_desc);
    std::cout << "\n=== Gate (a): multiplicity of minima ===\n";
    std::cout << "  restarts=" << result.all_restarts.size()
              << " clusters=" << gate_a.n_clusters
              << " labels=[";
    for (size_t i = 0; i < gate_a.labels.size(); ++i)
      std::cout << (i ? " " : "") << gate_a.labels[i];
    std::cout << "] passed=" << (gate_a.passed ? "yes" : "NO") << "\n";
    if (gate_a.passed)
      std::cout << "  ✓ Single basin of attraction\n";
    else
      std::cout << "  ✗ Multiple basins → personalisation BLOCKED\n";
  }

  // ---- Gate (b) ----
  {
    Params base_p = base_params;
    ObservationCost obs_cost(data, base_p, active_desc);

    auto gate_b = diagnose_structural_adequacy(
        data.cgm_time, data.cgm_value, data.meal_time,
        result.params.data(), obs_cost);

    std::cout << "\n=== Gate (b): structural adequacy ===\n";
    std::cout << "  Ljung-Box: Q=" << gate_b.ljung_box.Q
              << " p=" << gate_b.ljung_box.p_value
              << " lags=" << gate_b.ljung_box.n_lags
              << " significant=" << (gate_b.ljung_box.significant ? "yes" : "no") << "\n";
    for (auto& rs : gate_b.regimes) {
      std::cout << "    " << rs.name
                << ": n=" << rs.count
                << " bias=" << rs.bias_mgdl << " mg/dL"
                << " mean_glucose=" << rs.mean_glucose << "\n";
    }
    std::cout << "  Regime bias detected: " << (gate_b.regime_bias_detected ? "yes" : "no") << "\n";
    if (gate_b.passed)
      std::cout << "  ✓ No structural issues detected\n";
    else
      std::cout << "  " << gate_b.description << "\n";
  }

  // ---- Gate (c) ----
  {
    Params base_p = base_params;
    ObservationCost obs_cost(data, base_p, active_desc);

    auto gate_c = diagnose_observability(
        result.params.data(), obs_cost, active_desc);

    std::cout << "\n=== Gate (c): observability (SVD/FIM) ===\n";
    std::cout << "  Eigenvalues:";
    for (int i = 0; i < gate_c.n_params; ++i)
      std::cout << " " << gate_c.eigenvalues[i];
    std::cout << "\n";
    std::cout << "  Condition number: " << gate_c.condition_number << "\n";
    std::cout << "  Per-param labels:";
    for (auto& lbl : gate_c.param_labels)
      std::cout << " " << lbl;
    std::cout << "\n";
  }

  return 0;
}
