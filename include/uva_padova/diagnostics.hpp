#pragma once

#include <vector>
#include <string>
#include <Eigen/Core>
#include "uva_padova/optimizer.hpp"
#include "uva_padova/types.hpp"
#include "uva_padova/cost_function.hpp"

namespace uva_padova {

// ---------------------------------------------------------------------------
// DBSCAN clustering (simplified for small N)
// ---------------------------------------------------------------------------
struct DBSCANResult {
  std::vector<int> labels;  // per-point cluster ID, -1 = noise
  int n_clusters{0};
};

// Mahalanobis distance: d = sqrt((a-b)ᵀ Σ₀⁻¹ (a-b)).
// For diagonal Σ₀⁻¹ = diag(1/σᵢ²) this is simply:
//   d = sqrt( Σ (a[i]-b[i])² / σᵢ² )
DBSCANResult cluster_restarts(const std::vector<RestartRecord>& restarts,
                              const std::vector<ParamDescriptor>& desc,
                              double eps = 2.0,
                              int minPts = 2);

// ---------------------------------------------------------------------------
// Gate (a): multiplicity of minima detection
// ---------------------------------------------------------------------------
struct GateAResult {
  bool passed{false};
  int n_clusters{0};
  std::vector<int> labels;
};

// Diagnose whether CMA-ES restarts converged to a single basin of attraction.
// passed = true  ⇔ exactly one cluster with ≥ minPts points (others are noise).
// passed = false ⇔ ≥2 clusters → personalisation blocked.
GateAResult diagnose_multiplicity(const std::vector<RestartRecord>& restarts,
                                  const std::vector<ParamDescriptor>& desc,
                                  double eps = 2.0,
                                  int minPts = 2);

// ---------------------------------------------------------------------------
// Gate (b): structural adequacy — residual analysis
// ---------------------------------------------------------------------------
enum class RegimeType { Postprandial, Fasting };

struct RegimeLabel {
  RegimeType type;
  int meal_idx{-1};
};

struct RegimeStats {
  std::string name;
  int count{0};
  double mean_residual{0.0};
  double std_residual{0.0};
  double mean_glucose{0.0};
  double bias_mgdl{0.0};     // mean_residual, clipped
};

struct LjungBoxResult {
  double Q{0.0};
  double p_value{0.0};
  bool significant{false};
  int n_lags{0};
};

struct GateBResult {
  bool passed{false};
  LjungBoxResult ljung_box;
  std::vector<RegimeStats> regimes;
  bool regime_bias_detected{false};
  std::string description;
};

// Classify each CGM observation by regime (postprandial vs fasting)
std::vector<RegimeLabel> classify_regimes(
    const std::vector<double>& cgm_time,
    const std::vector<double>& meal_time,
    double window_min = 180.0);

// Ljung-Box test for autocorrelation in residuals
LjungBoxResult ljung_box_test(const std::vector<double>& residuals,
                               int max_lags = 20);

// Full structural adequacy diagnostic
GateBResult diagnose_structural_adequacy(
    const std::vector<double>& cgm_time,
    const std::vector<double>& cgm_value,
    const std::vector<double>& meal_time,
    const double* theta,
    const ObservationCost<>& obs_cost,
    double bias_threshold_mgdl = 10.0,
    double bias_threshold_pct = 0.05);

// ---------------------------------------------------------------------------
// Gate (c): observability via SVD of FIM_obs
// ---------------------------------------------------------------------------
struct GateCResult {
  int n_params{0};
  std::vector<double> eigenvalues;      // sorted descending
  double max_eigenvalue{0.0};
  double condition_number{0.0};
  std::vector<bool> prior_dominated;    // per-param
  std::vector<std::string> param_labels;
};

// Analyse observation Jacobian at theta. Parameters that project strongly
// onto weak eigen-directions (λ < weak_threshold · λ_max) are labelled
// "prior-dominated". Gate (c) never blocks — it only produces per-param labels
// for downstream use (CKF initialisation with expanded covariance).
GateCResult diagnose_observability(
    const double* theta,
    ObservationCost<>& obs_cost,
    const std::vector<ParamDescriptor>& desc,
    double weak_threshold = 1e-3);

} // namespace uva_padova
