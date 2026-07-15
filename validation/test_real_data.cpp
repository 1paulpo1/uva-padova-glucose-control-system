// =============================================================================
// Real-data identification pipeline.
//
// Loads CGM + basal CSV data for a single patient, estimates basal glucose,
// runs CMA-ES + LM identification, computes all three gates (A, B, C),
// and writes per-patient JSON output (compatible with cross_validate format).
//
// Usage:
//   ./test_real_data --cgm patient_cgm.csv --basal patient_basal.csv \
//                    [--json out.json] [--params-config config/...json] \
//                    [--restarts N] [--evals N] [--lm-iters N] \
//                    [--cmaes-lower L] [--cmaes-upper U] [--prior-std S]
//
// Ground-truth fields (theta_true, abs_err_log, ...) are set to null
// since real data has no known ground truth.
// =============================================================================

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>

#include <nlohmann/json.hpp>

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
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Helpers: optional<T> for CLI flags (NaN sentinel for doubles)
// ---------------------------------------------------------------------------
static double nan_dbl() { return std::numeric_limits<double>::quiet_NaN(); }

// ---------------------------------------------------------------------------
// JSON helpers (matching cross_validate.cpp format)
// ---------------------------------------------------------------------------
static json eigen_to_json(const Eigen::MatrixXd& m) {
  json j = json::array();
  for (int i = 0; i < m.rows(); ++i) {
    json row = json::array();
    for (int jj = 0; jj < m.cols(); ++jj)
      row.push_back(m(i, jj));
    j.push_back(std::move(row));
  }
  return j;
}

static json vec_to_json(const std::vector<double>& v) {
  json j = json::array();
  for (double x : v) j.push_back(x);
  return j;
}

static json null_or_vec(const std::vector<double>& v) {
  return v.empty() ? json(nullptr) : vec_to_json(v);
}

// ---------------------------------------------------------------------------
// Build per-patient JSON (subset of cross_validate's build_json_result;
// ground-truth fields set to null for real data)
// ---------------------------------------------------------------------------
static json build_json_result(int pid,
                              const IdentificationPipeline<>::Result& ires,
                              const std::vector<ParamDescriptor>& active_desc,
                              const Params& base_params,
                              const IdentificationData& data,
                              const GateAResult& ga,
                              const GateBResult& gb,
                              const GateCResult& gc,
                              const std::vector<double>& cgm_residuals,
                              double elapsed_s)
{
  json j;
  j["patient_id"] = pid;
  j["success"] = ires.success;
  j["cost"] = ires.cost;
  j["total_evals"] = ires.total_evals;
  j["best_cma_cost"] = ires.best_cma_cost;
  j["cmaes_evals"] = ires.cmaes_total_evals;
  j["lm_evals"] = ires.total_evals - ires.cmaes_total_evals;
  j["lm_n_iters"] = ires.lm_n_iters;
  j["lm_converged"] = ires.lm_converged;
  j["elapsed_s"] = elapsed_s;

  // Gate A
  j["gate_a"]["passed"] = ga.passed;
  j["gate_a"]["n_clusters"] = ga.n_clusters;
  {
    json lbl = json::array();
    for (int l : ga.labels) lbl.push_back(l);
    j["gate_a"]["labels"] = std::move(lbl);
  }

  // Gate B
  j["gate_b"]["passed"] = gb.passed;
  j["gate_b"]["lb_q"] = gb.ljung_box.Q;
  j["gate_b"]["lb_p_value"] = gb.ljung_box.p_value;
  j["gate_b"]["lb_significant"] = gb.ljung_box.significant;
  j["gate_b"]["lb_n_lags"] = gb.ljung_box.n_lags;
  j["gate_b"]["regime_bias"] = gb.regime_bias_detected;
  j["gate_b"]["description"] = gb.description;
  {
    json regimes = json::array();
    for (const auto& rs : gb.regimes) {
      json rj;
      rj["name"] = rs.name;
      rj["count"] = rs.count;
      rj["mean_residual"] = rs.mean_residual;
      rj["std_residual"] = rs.std_residual;
      rj["mean_glucose"] = rs.mean_glucose;
      rj["bias_mgdl"] = rs.bias_mgdl;
      regimes.push_back(std::move(rj));
    }
    j["gate_b"]["regimes"] = std::move(regimes);
  }

  // Gate C
  j["gate_c"]["n_prior_dominated"] = static_cast<int>(
      std::count(gc.prior_dominated.begin(), gc.prior_dominated.end(), true));
  j["gate_c"]["max_eigenvalue"] = gc.max_eigenvalue;
  j["gate_c"]["condition_number"] = gc.condition_number;
  {
    json ev = json::array();
    for (double v : gc.eigenvalues) ev.push_back(v);
    j["gate_c"]["eigenvalues"] = std::move(ev);
  }
  {
    json pl = json::array();
    for (const auto& s : gc.param_labels) pl.push_back(s);
    j["gate_c"]["param_labels"] = std::move(pl);
  }
  {
    json pd = json::array();
    for (bool b : gc.prior_dominated) pd.push_back(b);
    j["gate_c"]["prior_dominated"] = std::move(pd);
  }

  // Per-parameter recovery: ground truth is unknown for real data
  Params recovered_params;
  if (ires.success)
    recovered_params = ires.to_params(base_params, active_desc);

  json theta_rec = json::array();
  json rec_phys = json::array();
  for (size_t i = 0; i < active_desc.size(); ++i) {
    auto& name = active_desc[i].name;
    if (ires.success) {
      theta_rec.push_back(ires.params[i]);
      rec_phys.push_back(param_value(recovered_params, name));
    } else {
      theta_rec.push_back(nullptr);
      rec_phys.push_back(nullptr);
    }
  }
  j["theta_true"] = nullptr;
  j["theta_rec"] = std::move(theta_rec);
  j["abs_err_log"] = nullptr;
  j["abs_err_phys"] = nullptr;
  j["param_true_phys"] = nullptr;
  j["param_rec_phys"] = std::move(rec_phys);

  // CMA-ES restart details
  {
    json rst = json::array();
    for (const auto& rr : ires.all_restarts) {
      json rrj;
      rrj["cost"] = rr.cost;
      json rp = json::array();
      for (double v : rr.params) rp.push_back(v);
      rrj["params"] = std::move(rp);
      rst.push_back(std::move(rrj));
    }
    j["restarts"] = std::move(rst);
  }

  // Covariance matrix
  if (ires.cov.rows() > 0 && ires.cov.cols() > 0)
    j["cov"] = eigen_to_json(ires.cov);

  // CGM residuals
  j["cgm_residuals"] = null_or_vec(cgm_residuals);

  // Input data
  j["input"]["cgm_time"] = vec_to_json(data.cgm_time);
  j["input"]["cgm_value"] = vec_to_json(data.cgm_value);
  j["input"]["meal_time"] = vec_to_json(data.meal_time);
  j["input"]["meal_cho"] = vec_to_json(data.meal_cho);
  j["input"]["insulin_time"] = vec_to_json(data.insulin_time);
  j["input"]["insulin_rate"] = vec_to_json(data.insulin_rate);
  j["input"]["bolus_time"] = vec_to_json(data.bolus_time);
  j["input"]["bolus_dose"] = vec_to_json(data.bolus_dose);

  // No true_input for real data
  j["true_input"] = nullptr;

  return j;
}

// ===========================================================================
// Usage / help
// ===========================================================================
static void print_help(const char* prog) {
  std::cerr << "Usage: " << prog << " --cgm <csv> --basal <csv> [options]\n"
            << "\n"
            << "Required:\n"
            << "  --cgm FILE      CGM CSV (columns: t_min, cgm_mgdl, meal_mg, bolus_pmol_kg)\n"
            << "  --basal FILE    Basal CSV (columns: t_min, basal_u_per_hr)\n"
            << "\n"
            << "Output:\n"
            << "  --json FILE     Output JSON path (default: result.json)\n"
            << "\n"
            << "Parameter config:\n"
            << "  --params-config FILE  Parameter descriptor JSON (default: " UVA_PADOVA_CONFIG_DIR "/default_params.json)\n"
            << "\n"
            << "Optimizer:\n"
            << "  --restarts N    CMA-ES restarts (default: 3)\n"
            << "  --evals N       CMA-ES evaluations per restart (default: 500)\n"
            << "  --lm-iters N    Levenberg-Marquardt max iterations (default: 100)\n"
            << "  --cmaes-lower L CMA-ES lower bound (default: -3.0)\n"
            << "  --cmaes-upper U CMA-ES upper bound (default: 3.0)\n"
            << "  --prior-std S   Prior std override (default: from descriptor)\n"
            << "\n"
            << "Misc:\n"
            << "  --help          Show this message\n";
}

// ===========================================================================
int main(int argc, char* argv[]) {
  std::cout << std::fixed << std::setprecision(4);

  if (argc < 2) { print_help(argv[0]); return 1; }

  // ---- Parse CLI ----
  std::string cgm_csv, basal_csv, json_path = "result.json";
  std::string params_config_path;
  int restarts = -1;
  int evals = -1;
  int lm_iters = -1;
  double cmaes_lower = nan_dbl();
  double cmaes_upper = nan_dbl();
  double prior_std = nan_dbl();

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) {
        std::cerr << "ERROR: " << arg << " requires a value\n";
        std::exit(1);
      }
      return argv[i];
    };

    if (arg == "--cgm")           cgm_csv = next();
    else if (arg == "--basal")    basal_csv = next();
    else if (arg == "--json")     json_path = next();
    else if (arg == "--params-config") params_config_path = next();
    else if (arg == "--restarts") restarts = std::stoi(next());
    else if (arg == "--evals")    evals = std::stoi(next());
    else if (arg == "--lm-iters") lm_iters = std::stoi(next());
    else if (arg == "--cmaes-lower") cmaes_lower = std::stod(next());
    else if (arg == "--cmaes-upper") cmaes_upper = std::stod(next());
    else if (arg == "--prior-std")   prior_std = std::stod(next());
    else if (arg == "--help")     { print_help(argv[0]); return 0; }
    else {
      std::cerr << "WARNING: unknown flag " << arg << ", ignoring\n";
    }
  }

  if (cgm_csv.empty() || basal_csv.empty()) {
    std::cerr << "ERROR: --cgm and --basal are required\n";
    return 1;
  }

  int n_restarts = (restarts > 0) ? restarts : 3;
  int n_evals    = (evals > 0)    ? evals    : 500;
  int n_lm_iters = (lm_iters > 0) ? lm_iters : 100;
  double lower   = std::isnan(cmaes_lower) ? -3.0 : cmaes_lower;
  double upper   = std::isnan(cmaes_upper) ? 3.0  : cmaes_upper;
  double pstd    = std::isnan(prior_std)   ? 1.0  : prior_std;

  auto t_start = std::chrono::steady_clock::now();

  // ---- Base parameters ----
  Params base_params = Params::defaults_t1d();

  // ---- Load parameter descriptors ----
  std::string config_path = params_config_path.empty()
    ? UVA_PADOVA_CONFIG_DIR "/default_params.json"
    : params_config_path;

  auto all_desc = load_param_descriptors(config_path);
  auto active_desc = active_descriptors(all_desc);

  std::cout << "Config: " << config_path << "\n"
            << "  Params: " << active_desc.size() << " active\n";

  // ---- Load data (first pass) to estimate Gb from CGM ----
  auto data = RealDataReader::load_from_csv(cgm_csv, basal_csv, base_params);
  if (data.cgm_time.empty()) {
    std::cerr << "ERROR: failed to load data from " << cgm_csv << "\n";
    return 1;
  }

  std::cout << "Data: " << data.cgm_time.size() << " CGM, "
            << data.meal_time.size() << " meals, "
            << data.bolus_time.size() << " boluses, "
            << data.insulin_time.size() << " basal segments\n";

  // Estimate Gb from CGM median
  {
    auto sorted = data.cgm_value;
    std::sort(sorted.begin(), sorted.end());
    double Gb_est = sorted[sorted.size() / 2];
    std::cout << "Gb: " << Gb_est << " mg/dL (CGM median)\n";
    base_params.Gb = Gb_est;
    // Reload with adjusted Gb
    data = RealDataReader::load_from_csv(cgm_csv, basal_csv, base_params);
  }

  // Override prior_std if specified
  if (!std::isnan(prior_std)) {
    for (auto& d : active_desc)
      d.prior_std = pstd;
  }

  // ---- Run pipeline ----
  std::cout << "=== Identification ===\n";
  std::cout << "  Restarts: " << n_restarts
            << "  Evals: " << n_evals
            << "  LM iters: " << n_lm_iters
            << "  Bounds: [" << lower << ", " << upper << "]\n";

  auto result = IdentificationPipeline<>::run(
      data, base_params, active_desc,
      n_restarts, n_evals, n_lm_iters, lower, upper);

  std::cout << "Result: success=" << (result.success ? "YES" : "NO")
            << " cost=" << result.cost
            << " evals=" << result.total_evals << "\n";

  // ---- Gates ----
  GateAResult gateA;
  GateBResult gateB;
  GateCResult gateC;
  std::vector<double> cgm_residuals;

  if (result.success) {
    gateA = diagnose_multiplicity(result.all_restarts, active_desc);

    {
      ObservationCost<> obs_cost(data, base_params, active_desc);
      gateB = diagnose_structural_adequacy(
          data.cgm_time, data.cgm_value, data.meal_time,
          result.params.data(), obs_cost);
    }

    {
      ObservationCost<> obs_cost_gc(data, base_params, active_desc);
      gateC = diagnose_observability(result.params.data(), obs_cost_gc, active_desc);
    }

    // CGM residuals
    {
      ObservationCost<> obs_cost_res(data, base_params, active_desc);
      auto cgm_sim = obs_cost_res.simulate_observations<double>(result.params.data());
      cgm_residuals.resize(data.cgm_value.size());
      for (size_t i = 0; i < data.cgm_value.size(); ++i)
        cgm_residuals[i] = data.cgm_value[i] - cgm_sim[i];
    }
  }

  // ---- Console gate summary ----
  std::cout << "Gates: A=" << (gateA.passed ? "PASS" : "FAIL")
            << "(" << gateA.n_clusters << "cl)"
            << " B=" << (gateB.passed ? "PASS" : "FAIL")
            << "(" << (gateB.ljung_box.significant ? "autocorr " : "")
            << (gateB.regime_bias_detected ? "bias" : "") << ")"
            << " C=" << std::count(gateC.prior_dominated.begin(), gateC.prior_dominated.end(), true)
            << " dominated\n";

  // ---- Build and write JSON ----
  auto t_end = std::chrono::steady_clock::now();
  double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

  json jout = build_json_result(0, result, active_desc, base_params, data,
                                gateA, gateB, gateC, cgm_residuals, elapsed_s);

  // Top-level metadata
  jout["param_names"] = json::array();
  for (auto& d : active_desc)
    jout["param_names"].push_back(d.name);

  std::ofstream jfile(json_path);
  if (!jfile.is_open()) {
    std::cerr << "ERROR: cannot write " << json_path << "\n";
    return 1;
  }
  jfile << jout.dump(2) << "\n";
  jfile.close();

  std::cout << "Wrote " << json_path << " (" << elapsed_s << " s)\n";
  return 0;
}
