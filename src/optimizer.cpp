// IMPORTANT: include libcmaes BEFORE Ceres/glog to avoid LOG macro conflict
#include <libcmaes/cmaes.h>

#include "uva_padova/optimizer.hpp"
#include "uva_padova/config_loader.hpp"

#include <cmath>
#include <iostream>
#include <random>

#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace uva_padova {

// ===========================================================================
// CMA-ES wrapper
// ===========================================================================
CMAESOptimizer::Result CMAESOptimizer::run(
    std::function<double(const double*, int)> fit_func,
    const std::vector<double>& initial_sigma,
    int max_evals,
    int seed,
    double lower_bound,
    double upper_bound)
{
  int dim = static_cast<int>(initial_sigma.size());

  std::vector<double> sigma_vec = initial_sigma;

  if (seed < 0)
    seed = static_cast<int>(std::random_device{}());

  std::vector<double> x0(dim, 0.0);
  std::vector<double> lbounds(dim, lower_bound);
  std::vector<double> ubounds(dim, upper_bound);

  uint64_t rng_seed = static_cast<uint64_t>(seed);
  libcmaes::CMAParameters<> cmap(x0, sigma_vec, -1,
                                  lbounds, ubounds, rng_seed);
  cmap.set_algo(CMAES_DEFAULT);
  cmap.set_max_fevals(max_evals);

  // libcmaes NoBoundStrategy silently discards bounds — enforce them here
  // by wrapping the fit function with bounds checking
  libcmaes::FitFunc wrapped_func =
      [fit_func = std::move(fit_func), lbounds, ubounds](
          const double* x, const int& n) -> double {
    for (int i = 0; i < n; ++i)
      if (x[i] < lbounds[i] || x[i] > ubounds[i])
        return 1e20;
    return fit_func(x, n);
  };

  libcmaes::CMASolutions cmasols = libcmaes::cmaes(wrapped_func, cmap);

  auto bc = cmasols.best_candidate();
  Result res;
  res.best_cost = bc.get_fvalue();
  res.n_evals   = cmasols.fevals();
  const double* best_x = bc.get_x_ptr();
  res.best_params.assign(best_x, best_x + dim);
  return res;
}

// LMOptimizer::run and LMOptimizer::compute_covariance are templated on
// Simulator — definitions are in optimizer_impl.hpp (included from optimizer.hpp).

// ---------------------------------------------------------------------------
// Covariance + sandwich computation at arbitrary theta
// RETAINED FOR REFERENCE — unused in auto-mode (Ceres Covariance used instead)
// ---------------------------------------------------------------------------
LMOptimizer::CovarianceResult LMOptimizer::compute_covariance_at(
    const std::vector<double>& theta,
    ObservationCost<>& obs_cost,
    PriorCost& prior_cost)
{
  const int n_obs = obs_cost.n_obs_;
  const int n_params = obs_cost.n_params_;
  const int total_res = n_obs + prior_cost.n_params_;
  const double* params_ptr = theta.data();

  // Full Jacobian: observation + prior
  Eigen::MatrixXd J_full(total_res, n_params);

  // Observation Jacobian
  std::vector<double> obs_resid(n_obs);
  std::vector<double> obs_jac_buf(n_obs * n_params);
  double* obs_jac_ptrs[] = { obs_jac_buf.data() };
  obs_cost.Evaluate(&params_ptr, obs_resid.data(), obs_jac_ptrs);
  for (int i = 0; i < n_obs; ++i)
    for (int j = 0; j < n_params; ++j)
      J_full(i, j) = obs_jac_buf[i * n_params + j];

  // Prior Jacobian
  std::vector<double> prior_resid(prior_cost.n_params_);
  std::vector<double> prior_jac_buf(prior_cost.n_params_ * n_params);
  double* prior_jac_ptrs[] = { prior_jac_buf.data() };
  prior_cost.Evaluate(&params_ptr, prior_resid.data(), prior_jac_ptrs);
  for (int i = 0; i < prior_cost.n_params_; ++i)
    for (int j = 0; j < n_params; ++j)
      J_full(n_obs + i, j) = prior_jac_buf[i * n_params + j];

  // Naive posterior covariance: (J^T J + reg·I)⁻¹
  constexpr double kReg = 1e-12;
  Eigen::MatrixXd FIM = J_full.transpose() * J_full;
  Eigen::MatrixXd cov = (FIM + kReg * Eigen::MatrixXd::Identity(n_params, n_params)).inverse();

  // ---- Sandwich covariance: H⁻¹ · V · H⁻¹ ----
  // H = ∇²J_obs + Σ₀⁻¹  (Gauss-Newton: Σ ρ''(rᵢ)·∇rᵢ·∇rᵢᵀ + prior)
  // V = Σ ρ'(rᵢ)² · ∇rᵢ · ∇rᵢᵀ
  // cov_sandwich = H⁻¹ · V · H⁻¹
  std::vector<double> raw_resid;
  Eigen::MatrixXd J_raw;
  bool ok = obs_cost.compute_raw_gradients(theta.data(), raw_resid, J_raw);

  CovarianceResult result;
  if (!ok) {
    std::cerr << "  [sandwich] compute_raw_gradients failed (NaN in simulation); "
              << "returning prior-only covariance\n";
    result.cov = cov;
    result.cov_sandwich = cov;
    return result;
  }

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_params, n_params);
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n_params, n_params);

  ceres::SoftLOneLoss loss(1.0);
  double rho[3];

  for (int i = 0; i < n_obs; ++i) {
    double r   = raw_resid[i];
    double s   = r * r;
    loss.Evaluate(s, rho);
    double drho_dr   = rho[1] * 2.0 * r;
    double d2rho_dr2 = rho[1] * 2.0 + rho[2] * 4.0 * s;
    Eigen::Matrix<double, 1, Eigen::Dynamic> g = J_raw.row(i);
    H += d2rho_dr2 * g.transpose() * g;
    V += (drho_dr * drho_dr) * g.transpose() * g;
  }

  // Prior Hessian Σ₀⁻¹ — J_prior is diagonal diag(1/σᵢ)
  for (int i = 0; i < n_params; ++i) {
    double iv = 1.0 / (prior_cost.stds_[i] * prior_cost.stds_[i]);
    H(i, i) += iv;
  }

  Eigen::MatrixXd H_reg  = H + kReg * Eigen::MatrixXd::Identity(n_params, n_params);
  Eigen::MatrixXd H_inv  = H_reg.inverse();
  Eigen::MatrixXd cov_sandwich = H_inv * V * H_inv;

  result.cov          = std::move(cov);
  result.cov_sandwich = std::move(cov_sandwich);
  return result;
}

} // namespace uva_padova
