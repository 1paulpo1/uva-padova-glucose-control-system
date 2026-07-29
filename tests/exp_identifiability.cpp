// =============================================================================
// Identifiability experiment suite for the UVA-Padova model.
//
// Runs identification across multiple configurations and random seeds,
// computing per-parameter relative errors and posterior correlations.
// Output: CSV to stdout for downstream visualisation.
//
// NOTE: This test uses a 3-param subset (kp2, Vm0, p2U) for speed.
// The full 21-param configuration (all params except greedy-5 fixed set)
// is not represented by the current default test config.
// =============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "uva_padova/optimizer.hpp"
#include "uva_padova/constants.hpp"
#include "uva_padova/synthetic_data_generator.hpp"

using namespace uva_padova;

// ---------------------------------------------------------------------------
// Experiment configuration
// ---------------------------------------------------------------------------
struct ExperimentConfig {
  std::string label;
  int duration_min;
  bool with_bolus;
};

// ---------------------------------------------------------------------------
// Run a single identification trial, return metrics
// ---------------------------------------------------------------------------
struct TrialResult {
  double rel_err_kp2;
  double rel_err_Vm0;
  double rel_err_p2U;
  double corr_Vm0_p2U;
  double final_cost;
  bool success;
};

// RAII guard to suppress std::cout (for pipeline noise)
struct CoutSilencer {
  std::streambuf* old;
  std::ofstream null;
  CoutSilencer() : old(std::cout.rdbuf()), null("/dev/null") {
    if (null.is_open()) std::cout.rdbuf(null.rdbuf());
  }
  ~CoutSilencer() { std::cout.rdbuf(old); }
};

static TrialResult run_trial(const Params& true_p,
                             const ExperimentConfig& config,
                             int seed,
                             const std::vector<ParamDescriptor>& desc)
{
  Params base_p = Params::defaults_t1d();

  SyntheticDataConfig gen_cfg;
  gen_cfg.duration_min = config.duration_min;
  gen_cfg.cgm_interval = 5;
  gen_cfg.noise_sd = 3.0;
  gen_cfg.seed = seed;
  gen_cfg.meals = {{60, 60000}, {360, 80000}, {600, 70000}};
  if (config.with_bolus) {
    double bw = true_p.BW;
    gen_cfg.boluses = {
      {60,  6.0 * constants::kInsulinUnits_To_Pmol / bw},
      {360, 8.0 * constants::kInsulinUnits_To_Pmol / bw},
      {600, 7.0 * constants::kInsulinUnits_To_Pmol / bw},
    };
  }

  auto gen_result = generate_synthetic_data(true_p, base_p, desc, gen_cfg);
  auto& data = gen_result.data;

  CoutSilencer _silence;
  auto result = IdentificationPipeline<>::run(data, base_p, desc, 3, 500);

  Params recovered_p = result.to_params(base_p, desc);

  TrialResult tr;
  tr.rel_err_kp2 = std::abs(recovered_p.kp2 - true_p.kp2) / true_p.kp2;
  tr.rel_err_Vm0 = std::abs(recovered_p.Vm0 - true_p.Vm0) / true_p.Vm0;
  tr.rel_err_p2U = std::abs(recovered_p.p2U - true_p.p2U) / true_p.p2U;
  tr.final_cost  = result.cost;
  tr.success     = result.success;

  // Correlation Vm0 ↔ p2U from posterior covariance
  auto find_idx = [&](const std::string& name) -> int {
    for (size_t k = 0; k < desc.size(); ++k)
      if (desc[k].name == name) return static_cast<int>(k);
    return -1;
  };
  int i_vm0 = find_idx("Vm0");
  int i_p2u = find_idx("p2U");
  if (i_vm0 >= 0 && i_p2u >= 0 &&
      result.cov.rows() > std::max(i_vm0, i_p2u) &&
      result.cov.cols() > std::max(i_vm0, i_p2u)) {
    double cij = result.cov(i_vm0, i_p2u);
    double diag_i = result.cov(i_vm0, i_vm0);
    double diag_j = result.cov(i_p2u, i_p2u);
    if (diag_i > 0 && diag_j > 0) {
      tr.corr_Vm0_p2U = cij / std::sqrt(diag_i * diag_j);
    } else {
      tr.corr_Vm0_p2U = std::numeric_limits<double>::quiet_NaN();
      tr.success = false;
    }
  } else {
    tr.corr_Vm0_p2U = std::numeric_limits<double>::quiet_NaN();
    tr.success = false;
  }

  return tr;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
  std::cout << std::fixed << std::setprecision(6);

  // Ground truth
  Params true_p = Params::defaults_t1d();
  true_p.kp2  = 0.0030;
  true_p.Vm0  = 3.20;
  true_p.p2U  = 0.0280;

  // Parameter descriptors (must match order used in analysis)
  std::vector<ParamDescriptor> desc = {
    { "kp2", 0.0, 1.0, 0.0 },
    { "Vm0", 0.0, 1.0, 0.0 },
    { "p2U", 0.0, 1.0, 0.0 },
  };

  // Experiment configurations
  std::vector<ExperimentConfig> configs = {
    { "240_1meal",         240, false },
    { "720_3meals",        720, false },
    { "720_3meals_bolus",  720, true  },
  };

  // Random seeds
  std::vector<int> seeds = { 100, 200, 300, 400, 500, 600, 700, 800 };

  // CSV header
  std::cout << "config,seed,n_obs,rel_err_kp2,rel_err_Vm0,rel_err_p2U,"
            << "corr_Vm0_p2U,final_cost,success\n";

  for (const auto& cfg : configs) {
    // Summary accumulators
    std::vector<double> errs_kp2, errs_Vm0, errs_p2U, corrs;
    std::vector<int> n_obs_vec;

    for (int seed : seeds) {
      auto tr = run_trial(true_p, cfg, seed, desc);

      // Get n_obs from a quick data generation (only for CSV, not counted in trials)
      SyntheticDataConfig tmp_cfg;
      tmp_cfg.duration_min = cfg.duration_min;
      tmp_cfg.cgm_interval = 5;
      tmp_cfg.noise_sd = 3.0;
      tmp_cfg.seed = seed;
      tmp_cfg.meals = {{60, 60000}, {360, 80000}, {600, 70000}};
      auto tmp_gen = generate_synthetic_data(true_p, Params::defaults_t1d(), desc, tmp_cfg);
      auto& tmp_data = tmp_gen.data;

      std::cout << cfg.label << ","
                << seed << ","
                << tmp_data.cgm_time.size() << ","
                << tr.rel_err_kp2 << ","
                << tr.rel_err_Vm0 << ","
                << tr.rel_err_p2U << ","
                << tr.corr_Vm0_p2U << ","
                << tr.final_cost << ","
                << (tr.success ? "1" : "0") << "\n";

      errs_kp2.push_back(tr.rel_err_kp2);
      errs_Vm0.push_back(tr.rel_err_Vm0);
      errs_p2U.push_back(tr.rel_err_p2U);
      corrs.push_back(tr.corr_Vm0_p2U);
      n_obs_vec.push_back(static_cast<int>(tmp_data.cgm_time.size()));
    }

    auto mean = [](const std::vector<double>& v) {
      double s = 0; for (auto x : v) s += x; return s / v.size();
    };
    auto median = [](std::vector<double> v) {
      if (v.empty()) return 0.0;
      size_t n = v.size() / 2;
      std::nth_element(v.begin(), v.begin() + n, v.end());
      return v[n];
    };

    std::cerr << "\n--- " << cfg.label << " ---\n";
    std::cerr << "  obs: " << (n_obs_vec.empty() ? 0 : n_obs_vec[0]) << " per trial\n";
    std::cerr << "  kp2:  mean_err=" << mean(errs_kp2)  << " med_err=" << median(errs_kp2) << "\n";
    std::cerr << "  Vm0:  mean_err=" << mean(errs_Vm0)  << " med_err=" << median(errs_Vm0) << "\n";
    std::cerr << "  p2U:  mean_err=" << mean(errs_p2U)  << " med_err=" << median(errs_p2U) << "\n";
    std::cerr << "  corr(Vm0,p2U): mean=" << mean(corrs) << " med=" << median(corrs) << "\n";
  }

  return 0;
}
