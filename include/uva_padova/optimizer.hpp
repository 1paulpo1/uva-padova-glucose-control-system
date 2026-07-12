#pragma once

#include <vector>
#include <Eigen/Core>
#include "uva_padova/cost_function.hpp"

namespace uva_padova {

// ---------------------------------------------------------------------------
// CMA-ES optimizer wrapper (libcmaes)
// ---------------------------------------------------------------------------
struct CMAESOptimizer {
  struct Result {
    std::vector<double> best_params;
    double best_cost{1e20};
    int n_evals{0};
  };

  static Result run(const ObservationCost& cost,
                    const std::vector<double>& initial_sigma,
                    int max_evals = 1000,
                    int seed = -1);
};

// ---------------------------------------------------------------------------
// Ceres Levenberg-Marquardt optimizer
// ---------------------------------------------------------------------------
struct LMOptimizer {
  struct Result {
    std::vector<double> final_params;
    double final_cost{1e20};
    int n_iters{0};
    bool converged{false};
    Eigen::MatrixXd cov;           // posterior covariance via Ceres (loss-weighted)
    Eigen::MatrixXd cov_sandwich;  // retained for reference; unused in auto-mode
  };

  struct CovarianceResult {
    Eigen::MatrixXd cov;
    Eigen::MatrixXd cov_sandwich;
  };

  static Result run(const std::vector<double>& initial_params,
                    ObservationCost& obs_cost,
                    PriorCost& prior_cost,
                    int max_iters = 100);

  // Retained for reference — unused in auto-mode (Ceres Covariance used instead)
  static CovarianceResult compute_covariance_at(
      const std::vector<double>& theta,
      ObservationCost& obs_cost,
      PriorCost& prior_cost);
};

// ---------------------------------------------------------------------------
// Full pipeline: CMA-ES (×N restarts) → LM → top result
// ---------------------------------------------------------------------------
struct RestartRecord {
  std::vector<double> params;
  double cost{1e20};
};

struct IdentificationPipeline {
  struct Result {
    std::vector<double> params;
    double cost{1e20};
    int total_evals{0};
    bool success{false};
    Eigen::MatrixXd cov;           // posterior covariance from Ceres (loss-weighted)
    Eigen::MatrixXd cov_sandwich;  // retained for reference; unused in auto-mode
    std::vector<RestartRecord> all_restarts;  // best point from each CMA-ES restart

    Params to_params(const Params& base,
                     const std::vector<ParamDescriptor>& desc) const;
  };

  static Result run(const IdentificationData& data,
                    const Params& base_params,
                    const std::vector<ParamDescriptor>& param_desc,
                    int n_cmaes_restarts = 3,
                    int cmaes_evals = 1000,
                    int lm_max_iters = 100,
                    double cmaes_lower = -3.0,
                    double cmaes_upper = 3.0);

  // Convenience overload: load param descriptors from JSON config,
  // filter to only non-fixed params, and run the pipeline.
  // Returns the result AND the full param descriptor list (with fixed flags)
  // via the output parameter `full_desc`.
  static Result run_from_config(const IdentificationData& data,
                                const Params& base_params,
                                const std::string& json_path,
                                int n_cmaes_restarts = 3,
                                int cmaes_evals = 1000,
                                std::vector<ParamDescriptor>* full_desc = nullptr);
};

} // namespace uva_padova
