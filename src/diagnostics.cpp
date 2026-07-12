#include "uva_padova/diagnostics.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <boost/math/statistics/ljung_box.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <Eigen/Eigenvalues>

namespace uva_padova {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Mahalanobis distance for diagonal Σ₀⁻¹
static double mahalanobis_dist(const std::vector<double>& a,
                                const std::vector<double>& b,
                                const std::vector<double>& inv_sigma_sq)
{
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = a[i] - b[i];
    sum += d * d * inv_sigma_sq[i];
  }
  return std::sqrt(sum);
}

// Build inverse-variance vector from ParamDescriptor
static std::vector<double> build_inv_sigma_sq(const std::vector<ParamDescriptor>& desc)
{
  std::vector<double> inv_sq(desc.size());
  for (size_t i = 0; i < desc.size(); ++i) {
    double s = desc[i].prior_std;
    inv_sq[i] = 1.0 / (s * s);
  }
  return inv_sq;
}

// ---------------------------------------------------------------------------
// DBSCAN
// ---------------------------------------------------------------------------
DBSCANResult cluster_restarts(const std::vector<RestartRecord>& restarts,
                              const std::vector<ParamDescriptor>& desc,
                              double eps,
                              int minPts)
{
  int N = static_cast<int>(restarts.size());
  DBSCANResult result;
  result.labels.assign(N, -2);  // -2 = unvisited

  if (N == 0) {
    result.n_clusters = 0;
    return result;
  }

  auto inv_sigma_sq = build_inv_sigma_sq(desc);

  // Precompute pairwise distance matrix
  Eigen::MatrixXd D(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      D(i, j) = (i == j) ? 0.0
                         : mahalanobis_dist(restarts[i].params, restarts[j].params, inv_sigma_sq);

  // Find neighbourhood for each point
  std::vector<std::vector<int>> neighbors(N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i != j && D(i, j) <= eps)
        neighbors[i].push_back(j);
    }
  }

  // Core points: have ≥ minPts neighbours
  std::vector<bool> is_core(N, false);
  for (int i = 0; i < N; ++i) {
    if (static_cast<int>(neighbors[i].size()) >= std::max(0, minPts - 1))
      is_core[i] = true;
  }

  // Expand clusters
  int cluster_id = 0;
  for (int i = 0; i < N; ++i) {
    if (result.labels[i] != -2 || !is_core[i])
      continue;

    // Start a new cluster from this core point
    // BFS/DFS over density-reachable core points
    result.labels[i] = cluster_id;
    std::vector<int> stack = {i};

    while (!stack.empty()) {
      int p = stack.back();
      stack.pop_back();

      for (int q : neighbors[p]) {
        if (result.labels[q] == -2) {
          result.labels[q] = cluster_id;
          if (is_core[q])
            stack.push_back(q);
        }
      }
    }

    cluster_id++;
  }

  // Remaining unvisited points are noise
  for (int i = 0; i < N; ++i) {
    if (result.labels[i] == -2)
      result.labels[i] = -1;
  }

  // Noise label = -1 is not a cluster, so subtract it
  int noise_count = 0;
  for (int i = 0; i < N; ++i)
    if (result.labels[i] == -1)
      noise_count++;

  result.n_clusters = cluster_id;
  return result;
}

// ---------------------------------------------------------------------------
// Gate (a): multiplicity of minima
// ---------------------------------------------------------------------------
GateAResult diagnose_multiplicity(const std::vector<RestartRecord>& restarts,
                                  const std::vector<ParamDescriptor>& desc,
                                  double eps,
                                  int minPts)
{
  GateAResult gate;
  if (restarts.empty()) {
    gate.passed = false;
    return gate;
  }

  auto db = cluster_restarts(restarts, desc, eps, minPts);
  gate.labels     = std::move(db.labels);
  gate.n_clusters = db.n_clusters;

  // Passed only if exactly one cluster (noise is fine — single outlier restart
  // that didn't converge is allowed)
  gate.passed = (db.n_clusters == 1);

  return gate;
}

// ===========================================================================
//  Ljung-Box test (wraps Boost implementation)
// ===========================================================================
LjungBoxResult ljung_box_test(const std::vector<double>& residuals, int max_lags) {
  LjungBoxResult result;
  int n = static_cast<int>(residuals.size());
  if (n < 3) {
    result.significant = false;
    return result;
  }

  max_lags = std::min(max_lags, n - 1);
  if (max_lags < 1) {
    result.significant = false;
    return result;
  }

  auto [Q, p] = boost::math::statistics::ljung_box(residuals, max_lags, 0);

  result.Q = Q;
  result.p_value = p;
  result.n_lags = max_lags;
  result.significant = (p < 0.05);

  return result;
}

// ===========================================================================
//  Regime classification
// ===========================================================================
std::vector<RegimeLabel> classify_regimes(
    const std::vector<double>& cgm_time,
    const std::vector<double>& meal_time,
    double window_min)
{
  int n = static_cast<int>(cgm_time.size());
  std::vector<RegimeLabel> labels(n);

  for (int i = 0; i < n; ++i) {
    double t = cgm_time[i];
    labels[i].type = RegimeType::Fasting;
    labels[i].meal_idx = -1;

    for (int m = 0; m < static_cast<int>(meal_time.size()); ++m) {
      if (t >= meal_time[m] && t <= meal_time[m] + window_min) {
        labels[i].type = RegimeType::Postprandial;
        labels[i].meal_idx = m;
        break;
      }
    }
  }

  return labels;
}

// ===========================================================================
//  Gate (b): structural adequacy
// ===========================================================================
GateBResult diagnose_structural_adequacy(
    const std::vector<double>& cgm_time,
    const std::vector<double>& cgm_value,
    const std::vector<double>& meal_time,
    const double* theta,
    const ObservationCost<>& obs_cost,
    double bias_threshold_mgdl,
    double bias_threshold_pct)
{
  GateBResult gate;
  int n = static_cast<int>(cgm_time.size());

  // 1. Simulate G_sc at theta
  auto cgm_sim = obs_cost.simulate_observations<double>(theta);

  // 2. Compute residuals
  std::vector<double> residuals(n);
  for (int i = 0; i < n; ++i)
    residuals[i] = cgm_value[i] - cgm_sim[i];

  // 3. Ljung-Box autocorrelation test
  int max_lags = std::min(20, n / 5);
  gate.ljung_box = ljung_box_test(residuals, max_lags);

  // 4. Classify regimes
  auto labels = classify_regimes(cgm_time, meal_time, 180.0);

  // 5. Per-regime statistics
  struct Accum {
    double sum_res{0.0};
    double sum_res_sq{0.0};
    double sum_glu{0.0};
    int count{0};
  };
  Accum reg_acc[2];  // 0=Fasting, 1=Postprandial
  double regime_glucose_sums[2] = {0.0, 0.0};
  int regime_counts[2] = {0, 0};

  for (int i = 0; i < n; ++i) {
    int ri = (labels[i].type == RegimeType::Postprandial) ? 1 : 0;
    reg_acc[ri].sum_res   += residuals[i];
    reg_acc[ri].sum_res_sq += residuals[i] * residuals[i];
    reg_acc[ri].sum_glu   += cgm_value[i];
    reg_acc[ri].count++;
  }

  gate.regimes.resize(2);
  gate.regimes[0].name = "Fasting";
  gate.regimes[1].name = "Postprandial";

  for (int ri = 0; ri < 2; ++ri) {
    auto& rs = gate.regimes[ri];
    auto& acc = reg_acc[ri];
    rs.count = acc.count;

    if (acc.count > 0) {
      rs.mean_residual = acc.sum_res / acc.count;
      rs.mean_glucose  = acc.sum_glu / acc.count;
      double variance = (acc.sum_res_sq - acc.sum_res * acc.sum_res / acc.count)
                        / (acc.count - 1);
      rs.std_residual = (variance > 0) ? std::sqrt(variance) : 0.0;
      rs.bias_mgdl = rs.mean_residual;

      // Two-sample? No — one-sample t-test: H₀: mean=0
      if (rs.std_residual > 1e-9 && acc.count > 1) {
        double t_stat = rs.mean_residual / (rs.std_residual / std::sqrt(acc.count));
        double p_val = 2.0 * boost::math::cdf(
            boost::math::complement(boost::math::students_t(acc.count - 1),
                                    std::abs(t_stat)));

        // Effect size threshold: max(10 mg/dL, 5% of mean glucose)
        double eff_thresh = std::max(bias_threshold_mgdl,
                                     bias_threshold_pct * rs.mean_glucose);

        if (p_val < 0.05 && std::abs(rs.mean_residual) > eff_thresh) {
          gate.regime_bias_detected = true;
        }
      }
    }
  }

  // 6. Overall verdict
  gate.passed = !gate.ljung_box.significant && !gate.regime_bias_detected;

  // 7. Description
  if (gate.passed) {
    gate.description = "No structural issues detected";
  } else {
    std::string reasons;
    if (gate.ljung_box.significant)
      reasons += "autocorrelation (Q=" + std::to_string(gate.ljung_box.Q)
                 + " p=" + std::to_string(gate.ljung_box.p_value) + ") ";
    if (gate.regime_bias_detected)
      reasons += "regime bias ";
    gate.description = "FAILED: " + reasons;
  }

  return gate;
}

// ===========================================================================
//  Gate (c): observability via SVD of FIM_obs
// ===========================================================================
GateCResult diagnose_observability(
    const double* theta,
    ObservationCost<>& obs_cost,
    const std::vector<ParamDescriptor>& desc,
    double weak_threshold)
{
  GateCResult gate;
  int n_params = obs_cost.n_params_;
  gate.n_params = n_params;

  if (n_params == 0) return gate;

  // 1. Compute raw (normalised) Jacobian J_r at theta,
  //    where r_i = (G_sim_i - G_obs_i) / σ.
  //
  //    We use compute_raw_gradients (retained for diagnostics); with plain
  //    residuals in Evaluate() both methods return the same Jacobian.
  std::vector<double> raw_resid;
  Eigen::MatrixXd J_raw;
  bool ok = obs_cost.compute_raw_gradients(theta, raw_resid, J_raw);
  if (!ok) {
    std::cerr << "  [GateC] compute_raw_gradients failed (NaN in simulation)"
              << " — marking all params as prior-dominated\n";
    gate.param_labels.reserve(n_params);
    gate.prior_dominated.assign(n_params, true);
    gate.max_eigenvalue = 0.0;
    gate.condition_number = 0.0;
    for (int i = 0; i < n_params; ++i) {
      gate.param_labels.push_back(desc[i].name);
      gate.eigenvalues.push_back(0.0);
    }
    return gate;
  }

  // 2. FIM_obs = J_raw^T J_raw  (normalised observed Fisher Information)
  Eigen::MatrixXd FIM = (J_raw.transpose() * J_raw).eval();

  // 3. Eigendecomposition (SelfAdjointEigenSolver sorts ascending)
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(FIM);
  gate.eigenvalues.resize(n_params);
  for (int i = 0; i < n_params; ++i) {
    gate.eigenvalues[i] = eig.eigenvalues()(n_params - 1 - i);
  }
  gate.max_eigenvalue = gate.eigenvalues[0];
  double min_ev = gate.eigenvalues[n_params - 1];
  gate.condition_number = (min_ev > 0.0)
      ? gate.max_eigenvalue / min_ev
      : std::numeric_limits<double>::infinity();

  // 4. Weak directions: λ_i < weak_threshold · λ_max
  //    Eigenvectors in eig.eigenvectors().col(j) correspond to
  //    eig.eigenvalues()(j) (ascending).  The smallest eigenvalues
  //    sit at j = 0, 1, …, so we iterate from j = 0 upward.
  double thr = weak_threshold * gate.max_eigenvalue;
  gate.prior_dominated.assign(n_params, false);
  for (int j = 0; j < n_params; ++j) {
    double ev = eig.eigenvalues()(j);
    if (ev >= thr) break;
    for (int p = 0; p < n_params; ++p) {
      // Heuristic: if |v_p| > 0.5, parameter p dominates (>25% energy)
      // in this weak direction.  Multiple parameters may simultaneously
      // exceed 0.5 from the same weak direction — this is expected
      // behaviour (two parameters can jointly dominate a weak direction),
      // not a bug.
      if (!gate.prior_dominated[p] &&
          std::abs(eig.eigenvectors().col(j)(p)) > 0.5) {
        gate.prior_dominated[p] = true;
      }
    }
  }

  // 5. Per-parameter labels
  gate.param_labels.resize(n_params);
  for (int i = 0; i < n_params; ++i) {
    std::string name = (i < static_cast<int>(desc.size())) ? desc[i].name : "?";
    gate.param_labels[i] = name
        + (gate.prior_dominated[i] ? "=prior-dominated" : "=data-informed");
  }

  return gate;
}

} // namespace uva_padova
