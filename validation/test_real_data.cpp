// =============================================================================
// Test the identification pipeline on real patient data.
//
// Usage:
//     ./test_real_data <cgm_csv> <basal_csv> [n_restarts] [cmaes_evals] [mode]
//     mode: '3' (kp2+Vm0+p2U, default) or '6' (+ka1+ka2+kd)
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

#include "csv_reader.hpp"

using namespace uva_padova;

int main(int argc, char* argv[]) {
  std::cout << std::scientific << std::setprecision(6);

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <cgm_csv> <basal_csv> [n_restarts] [cmaes_evals] [mode]\n";
    std::cerr << "  mode: '3' (default, kp2+Vm0+p2U) or '6' (+ka1+ka2+kd)\n";
    return 1;
  }

  std::string cgm_csv = argv[1];
  std::string basal_csv = argv[2];
  int n_restarts = (argc > 3) ? std::atoi(argv[3]) : 3;
  int cmaes_evals = (argc > 4) ? std::atoi(argv[4]) : 500;
  std::string mode = (argc > 5) ? argv[5] : "3";

  // ---- Choose params to identify ----
  std::vector<ParamDescriptor> desc;
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
  std::cout << "  Mode: " << desc.size() << " params (";
  for (size_t i = 0; i < desc.size(); ++i)
    std::cout << (i ? ", " : "") << desc[i].name;
  std::cout << ")\n";

  Params base_params = Params::defaults_t1d();
  // Use patient-specific weight from CSV if available (hardcoded for now)
  // Patient 563 weight = 99 kg
  base_params.BW = 99.0;

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
  auto result = IdentificationPipeline::run(data, base_params, desc,
                                            n_restarts, cmaes_evals);

  // ---- Print result ----
  std::cout << "\n=== Pipeline result ===\n";
  std::cout << "  cost=" << result.cost << "\n";
  std::cout << "  success=" << (result.success ? "yes" : "no") << "\n";
  std::cout << "  theta = [";
  for (size_t i = 0; i < result.params.size(); ++i)
    std::cout << (i ? ", " : "") << result.params[i];
  std::cout << "]\n";

  auto recovered = result.to_params(base_params, desc);
  auto default_p = Params::defaults_t1d();
  std::cout << "  Recovered kp2 = " << recovered.kp2
            << " (default=" << default_p.kp2 << ")\n";
  std::cout << "  Recovered Vm0 = " << recovered.Vm0
            << " (default=" << default_p.Vm0 << ")\n";
  std::cout << "  Recovered p2U = " << recovered.p2U
            << " (default=" << default_p.p2U << ")\n";
  if (mode == "6" || mode == "9") {
    std::cout << "  Recovered ka1 = " << recovered.ka1
              << " (default=" << default_p.ka1 << ")\n";
    std::cout << "  Recovered ka2 = " << recovered.ka2
              << " (default=" << default_p.ka2 << ")\n";
    std::cout << "  Recovered kd  = " << recovered.kd
              << " (default=" << default_p.kd << ")\n";
  }
  if (mode == "9") {
    std::cout << "  Recovered kabs = " << recovered.kabs
              << " (default=" << default_p.kabs << ")\n";
    std::cout << "  Recovered Gb   = " << recovered.Gb
              << " (estimated from CGM=" << base_params.Gb << ")\n";
    std::cout << "  Recovered VG   = " << recovered.VG
              << " (default=" << default_p.VG << ")\n";
  }

  // ---- Posterior cov ----
  std::cout << "\n=== Posterior cov (from pipeline) ===\n";
  if (result.cov.rows() > 0) {
    std::cout << result.cov << "\n";
    auto print_corr = [&](int i, int j) {
      double c = result.cov(i,j) / std::sqrt(result.cov(i,i) * result.cov(j,j));
      std::cout << "  corr(" << desc[i].name << "," << desc[j].name << ")=" << c << "\n";
    };
    for (size_t i = 0; i < desc.size(); ++i)
      for (size_t j = i+1; j < desc.size(); ++j)
        print_corr(static_cast<int>(i), static_cast<int>(j));
  } else {
    std::cout << "  (empty)\n";
  }

  // ---- Gate (a) ----
  {
    auto gate_a = diagnose_multiplicity(result.all_restarts, desc);
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
    ObservationCost obs_cost(data, base_p, desc);

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
    ObservationCost obs_cost(data, base_p, desc);

    auto gate_c = diagnose_observability(
        result.params.data(), obs_cost, desc);

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
