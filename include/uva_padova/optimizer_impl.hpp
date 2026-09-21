#pragma once

#include "uva_padova/config_loader.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include <ceres/ceres.h>
#include <Eigen/Dense>

namespace uva_padova {

// ===========================================================================
// Helper: validate that the Jacobian of obs_cost at theta is all-finite.
// Returns true if covariance computation is safe.
// ===========================================================================
namespace detail {
template <typename Simulator>
static bool validate_jacobian(ObservationCost<Simulator>& obs_cost,
                              const std::vector<double>& theta) {
  if (theta.empty()) return false;
  int n_obs = obs_cost.n_obs_;
  int n_params = obs_cost.n_params_;
  const double* params_ptr = theta.data();
  std::vector<double> resid(n_obs);
  std::vector<double> jac_buf(n_obs * n_params);
  double* jac_ptrs[] = { jac_buf.data() };
  if (!obs_cost.Evaluate(&params_ptr, resid.data(), jac_ptrs))
    return false;
  for (int i = 0; i < n_obs * n_params; ++i)
    if (!std::isfinite(jac_buf[i])) return false;
  return true;
}

inline Eigen::MatrixXd prior_covariance(const PriorCost& prior_cost) {
  const int dim = prior_cost.n_params_;
  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(dim, dim);
  for (int i = 0; i < dim; ++i)
    cov(i, i) = prior_cost.stds_[i] * prior_cost.stds_[i];
  return cov;
}
} // namespace detail

// ===========================================================================
// Ceres Levenberg-Marquardt optimizer (templated on Simulator)
// ===========================================================================
template <typename Simulator>
LMOptimizer::Result LMOptimizer::run(
    const std::vector<double>& initial_params,
    ObservationCost<Simulator>& obs_cost,
    PriorCost& prior_cost,
    int max_iters)
{
  std::vector<double> x = initial_params;

  ceres::Problem::Options problem_options;
  problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  auto cache = std::make_shared<SimulationCache>();
  std::vector<std::unique_ptr<PerObservationCost<Simulator>>> per_costs;
  per_costs.reserve(obs_cost.n_obs_);
  for (int i = 0; i < obs_cost.n_obs_; ++i) {
    per_costs.push_back(std::make_unique<PerObservationCost<Simulator>>(
        &obs_cost, i, cache));
    problem.AddResidualBlock(per_costs.back().get(),
                             new ceres::SoftLOneLoss(1.0), x.data());
  }
  problem.AddResidualBlock(&prior_cost, nullptr, x.data());

  for (int i = 0; i < obs_cost.n_params_; ++i) {
    const auto& pd = obs_cost.param_desc_[i];
    const double lower = pd.prior_mean - 3.0 * pd.prior_std;
    const double upper = pd.prior_mean + 3.0 * pd.prior_std;
    problem.SetParameterLowerBound(x.data(), i, lower);
    problem.SetParameterUpperBound(x.data(), i, upper);
  }

  ceres::Solver::Options options;
  options.minimizer_type = ceres::TRUST_REGION;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = max_iters;
  options.function_tolerance = 1e-8;
  options.gradient_tolerance = 1e-10;
  options.parameter_tolerance = 1e-8;
  options.initial_trust_region_radius = 1e2;   // τ = 1e-2, Madsen-Nielsen-Tingleff §6.2
  options.max_trust_region_radius = 1e6;       // prevents runaway radius growth
  options.minimizer_progress_to_stdout = true;
  options.logging_type = ceres::PER_MINIMIZER_ITERATION;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Eigen::MatrixXd cov_matrix = detail::prior_covariance(prior_cost);
  {
    // Do not enter the covariance path when Ceres already declared the
    // solution unusable. In that state SuiteSparse/CHOLMOD may fail during
    // factorisation even if a direct Jacobian probe later looks finite.
    bool jacobian_ok = detail::validate_jacobian(obs_cost, x);
    if (jacobian_ok && summary.IsSolutionUsable()) {
      ceres::Covariance::Options cov_options;
      cov_options.apply_loss_function = true;
      ceres::Covariance covariance(cov_options);
      std::vector<std::pair<const double*, const double*>> cov_blocks;
      cov_blocks.push_back({x.data(), x.data()});
      if (covariance.Compute(cov_blocks, &problem)) {
        covariance.GetCovarianceBlock(x.data(), x.data(), cov_matrix.data());
      }
    }
  }

  std::cout << "  LM: " << summary.BriefReport() << "\n";

  Result res;
  res.final_params = x;
  res.n_iters      = static_cast<int>(summary.iterations.size());
  res.converged    = summary.termination_type == ceres::CONVERGENCE;
  res.final_cost   = summary.final_cost;
  res.cov          = std::move(cov_matrix);

  return res;
}

template <typename Simulator>
LMOptimizer::CovarianceResult LMOptimizer::compute_covariance(
    const std::vector<double>& theta,
    ObservationCost<Simulator>& obs_cost,
    PriorCost& prior_cost)
{
  int dim = static_cast<int>(theta.size());
  if (dim == 0) return {};

  std::vector<double> x = theta;

  ceres::Problem::Options tmp_problem_options;
  tmp_problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem tmp_problem(tmp_problem_options);
  auto cache = std::make_shared<SimulationCache>();
  std::vector<std::unique_ptr<PerObservationCost<Simulator>>> per_costs;
  per_costs.reserve(obs_cost.n_obs_);
  for (int i = 0; i < obs_cost.n_obs_; ++i) {
    per_costs.push_back(std::make_unique<PerObservationCost<Simulator>>(
        &obs_cost, i, cache));
    tmp_problem.AddResidualBlock(per_costs.back().get(),
                                 new ceres::SoftLOneLoss(1.0), x.data());
  }
  tmp_problem.AddResidualBlock(&prior_cost, nullptr, x.data());

  Eigen::MatrixXd cov_mat = detail::prior_covariance(prior_cost);
  bool jacobian_ok = detail::validate_jacobian(obs_cost, x);
  if (jacobian_ok) {
    ceres::Covariance::Options cov_opts;
    cov_opts.apply_loss_function = true;
    ceres::Covariance tmp_cov(cov_opts);
    std::vector<std::pair<const double*, const double*>> cb = {
        {x.data(), x.data()}};
    if (tmp_cov.Compute(cb, &tmp_problem)) {
      cov_mat.resize(dim, dim);
      tmp_cov.GetCovarianceBlock(x.data(), x.data(), cov_mat.data());
    }
  }

  return {cov_mat, {}};
}

template <typename Simulator>
LMOptimizer::CovarianceResult LMOptimizer::compute_covariance_at(
    const std::vector<double>& theta,
    ObservationCost<Simulator>& obs_cost,
    PriorCost& prior_cost)
{
  const int n_obs = obs_cost.n_obs_;
  const int n_params = obs_cost.n_params_;
  const int total_res = n_obs + prior_cost.n_params_;
  const double* params_ptr = theta.data();

  Eigen::MatrixXd J_full(total_res, n_params);

  std::vector<double> obs_resid(n_obs);
  std::vector<double> obs_jac_buf(n_obs * n_params);
  double* obs_jac_ptrs[] = { obs_jac_buf.data() };
  obs_cost.Evaluate(&params_ptr, obs_resid.data(), obs_jac_ptrs);
  for (int i = 0; i < n_obs; ++i)
    for (int j = 0; j < n_params; ++j)
      J_full(i, j) = obs_jac_buf[i * n_params + j];

  std::vector<double> prior_resid(prior_cost.n_params_);
  std::vector<double> prior_jac_buf(prior_cost.n_params_ * n_params);
  double* prior_jac_ptrs[] = { prior_jac_buf.data() };
  prior_cost.Evaluate(&params_ptr, prior_resid.data(), prior_jac_ptrs);
  for (int i = 0; i < prior_cost.n_params_; ++i)
    for (int j = 0; j < n_params; ++j)
      J_full(n_obs + i, j) = prior_jac_buf[i * n_params + j];

  constexpr double kReg = 1e-12;
  Eigen::MatrixXd FIM = J_full.transpose() * J_full;
  Eigen::MatrixXd cov = (FIM + kReg * Eigen::MatrixXd::Identity(n_params, n_params)).inverse();

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

  // NOTE: uses per-observation loss as a robust working model for the
  // sandwich estimator. This differs from the aggregate SoftLOneLoss used
  // in the optimization objective (compute_scalar_cost and Ceres), which
  // is intentional — the sandwich estimator is designed to be robust to
  // working model misspecification, and per-observation loss yields a
  // full-rank V matrix (aggregate loss would give rank-1 V).
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

// ===========================================================================
// Pipeline: CMA-ES (×N restarts) → LM → top result
// ===========================================================================
template <typename CMAES, typename LM>
auto IdentificationPipeline<CMAES, LM>::run(
    const IdentificationData& data,
    const Params& base_params,
    const std::vector<ParamDescriptor>& param_desc,
    int n_cmaes_restarts,
    int cmaes_evals,
    int lm_max_iters,
    double cmaes_lower,
    double cmaes_upper) -> Result
{
  int dim = static_cast<int>(param_desc.size());

  ObservationCost<> obs_cost(data, base_params, param_desc);
  PriorCost prior_cost(param_desc);

  std::vector<double> initial_sigma(dim);
  std::vector<double> lower_bounds(dim);
  std::vector<double> upper_bounds(dim);
  for (int i = 0; i < dim; ++i)
    initial_sigma[i] = param_desc[i].prior_std;
  for (int i = 0; i < dim; ++i) {
    lower_bounds[i] = param_desc[i].prior_mean + cmaes_lower * param_desc[i].prior_std;
    upper_bounds[i] = param_desc[i].prior_mean + cmaes_upper * param_desc[i].prior_std;
  }

  double best_cma_cost = 1e20;
  std::vector<double> best_cma_point(dim, 0.0);
  std::vector<RestartRecord> restart_points;
  restart_points.reserve(n_cmaes_restarts);

  std::function<double(const double*, int)> fit_func =
      [&obs_cost, &prior_cost](const double* x, int n) -> double {
    (void)n;
    double fval = obs_cost.compute_scalar_cost(x) + prior_cost.compute_scalar_prior(x);
    if (!std::isfinite(fval)) return 1e20;
    return fval;
  };

  int cmaes_total_evals = 0;
  for (int restart = 0; restart < n_cmaes_restarts; ++restart) {
    int seed = static_cast<int>(std::random_device{}());
    typename CMAES::Result cmaes_res;
    try {
      cmaes_res = CMAES::run(fit_func, initial_sigma, cmaes_evals,
                             seed, lower_bounds, upper_bounds);
    } catch (const std::exception& e) {
      std::cerr << "  CMA-ES restart " << (restart + 1) << "/"
                << n_cmaes_restarts << " threw: " << e.what() << "\n";
      cmaes_res = {{}, 1e20, 0};
    }

    if (cmaes_res.best_cost < best_cma_cost) {
      best_cma_cost = cmaes_res.best_cost;
      best_cma_point = cmaes_res.best_params;
    }

    restart_points.push_back({cmaes_res.best_params, cmaes_res.best_cost});
    cmaes_total_evals += cmaes_res.n_evals;

    std::cout << "  CMA-ES restart " << (restart + 1) << "/"
              << n_cmaes_restarts << ": cost=" << cmaes_res.best_cost
              << " fevals=" << cmaes_res.n_evals << "\n";
  }

  auto lm_res = LM::run(best_cma_point, obs_cost, prior_cost, lm_max_iters);

  Result result;
  bool cmaes_failed = (best_cma_cost >= 1e10);
  bool lm_valid = std::isfinite(lm_res.final_cost) && lm_res.final_cost >= 0;
  if (lm_valid && lm_res.final_cost < best_cma_cost && (cmaes_failed || lm_res.converged)) {
    std::cout << "  LM improved cost: " << lm_res.final_cost
              << " (iters=" << lm_res.n_iters << ")\n";
    result.params  = lm_res.final_params;
    result.cost    = lm_res.final_cost;
    result.success = true;
  } else if (cmaes_failed) {
    std::cout << "  CMA-ES and LM both failed; marking as failure\n";
    result.params  = best_cma_point;
    result.cost    = best_cma_cost;
    result.success = false;
  } else {
    std::cout << "  LM did not improve; using CMA-ES best point\n";
    result.params  = best_cma_point;
    result.cost    = best_cma_cost;
    result.success = true;
  }
  result.total_evals = obs_cost.eval_count();
  result.all_restarts = std::move(restart_points);
  result.best_cma_cost = best_cma_cost;
  result.cmaes_total_evals = cmaes_total_evals;
  result.lm_n_iters = lm_res.n_iters;
  result.lm_converged = lm_res.converged;

  if (result.success && !result.params.empty()) {
    int dim = obs_cost.n_params_;
    auto cov_sandwich_res = LM::compute_covariance_at(result.params, obs_cost, prior_cost);
    if (lm_res.cov.rows() == dim && result.params == lm_res.final_params) {
      result.cov = lm_res.cov;
    } else {
      auto cov_res = LM::compute_covariance(result.params, obs_cost, prior_cost);
      result.cov = cov_res.cov;
    }
    result.cov_sandwich = std::move(cov_sandwich_res.cov_sandwich);
    std::cout << "  [debug] covariance done\n";
  } else {
    result.cov_sandwich = Eigen::MatrixXd{};
  }

  return result;
}

template <typename CMAES, typename LM>
auto IdentificationPipeline<CMAES, LM>::run_from_config(
    const IdentificationData& data,
    const Params& base_params,
    const std::string& json_path,
    int n_cmaes_restarts,
    int cmaes_evals,
    std::vector<ParamDescriptor>* full_desc) -> Result
{
  auto all = load_param_descriptors(json_path);
  if (all.empty()) {
    std::cerr << "pipeline: failed to load config from " << json_path << "\n";
    Result err;
    err.success = false;
    return err;
  }

  if (full_desc)
    *full_desc = all;

  auto active = active_descriptors(all);
  std::cout << "  Config: " << all.size() << " params total, "
            << active.size() << " active ("
            << (all.size() - active.size()) << " fixed)\n";

  return run(data, base_params, active, n_cmaes_restarts, cmaes_evals);
}

template <typename CMAES, typename LM>
Params IdentificationPipeline<CMAES, LM>::Result::to_params(
    const Params& base,
    const std::vector<ParamDescriptor>& desc) const
{
  Params p = base;
  size_t theta_idx = 0;
  for (size_t i = 0; i < desc.size(); ++i) {
    if (desc[i].fixed) continue;
    if (theta_idx >= params.size()) break;
    double mult = std::exp(params[theta_idx]);
    if (!std::isfinite(mult)) mult = 1.0;
    param_by_id(p, param_id_from_string(desc[i].name)) *= mult;
    ++theta_idx;
  }
  return p;
}

} // namespace uva_padova
