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
    const ObservationCost& obs_cost,
    const std::vector<double>& initial_sigma,
    int max_evals,
    int seed)
{
  // Use PriorCost to compute MAP prior for CMA-ES
  // We build a dummy descriptor vector with matching size
  // (the actual prior comes from the pipeline's PriorCost)
  int dim = obs_cost.n_params_;

  // For CMA-ES we need the full scalar cost = observation + prior.
  // PriorCost is not available here — the pipeline passes it via the fit lambda.
  // We handle this differently: the pipeline's lambda calls both.

  std::vector<double> x0(dim, 0.0);

  std::vector<double> sigma_vec = initial_sigma;
  if (sigma_vec.empty())
    sigma_vec.assign(dim, 1.0);

  uint64_t rng_seed = (seed >= 0) ? static_cast<uint64_t>(seed)
                                   : static_cast<uint64_t>(std::random_device{}());
  std::vector<double> empty_bounds;
  libcmaes::CMAParameters<> cmap(x0, sigma_vec, -1, empty_bounds, empty_bounds, rng_seed);
  cmap.set_algo(CMAES_DEFAULT);
  cmap.set_max_fevals(max_evals);

  // The fit function is passed as a lambda from the pipeline
  // This run() function is only called with the lambda from IdentificationPipeline::run
  // which wraps both obs_cost and prior_cost

  // Note: this is only called from the pipeline, which uses a custom FitFunc lambda.
  // The default implementation here is not used directly.
  Result res;
  res.best_cost = 1e20;
  return res;
}

// ===========================================================================
// Ceres Levenberg-Marquardt optimizer
// ===========================================================================
LMOptimizer::Result LMOptimizer::run(
    const std::vector<double>& initial_params,
    ObservationCost& obs_cost,
    PriorCost& prior_cost,
    int max_iters)
{
  std::vector<double> x = initial_params;

  // Build Ceres problem
  ceres::Problem::Options problem_options;
  problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  // Observation residuals — log-cosh robustness is embedded in residuals
  // via sqrt-trick: residual = sign(r)·√(2·log(cosh(r))), r = (G-obs)/σ
  // so ½·Σ s² = Σ log(cosh(r)) without needing a separate LossFunction.
  problem.AddResidualBlock(&obs_cost, nullptr, x.data());

  // MAP prior residuals (quadratic)
  problem.AddResidualBlock(&prior_cost, nullptr, x.data());

  // Parameter bounds (θ ∈ [-3, 3] in log-space)
  for (int i = 0; i < obs_cost.n_params_; ++i) {
    problem.SetParameterLowerBound(x.data(), i, -3.0);
    problem.SetParameterUpperBound(x.data(), i, 3.0);
  }

  // Solver options
  ceres::Solver::Options options;
  options.minimizer_type = ceres::TRUST_REGION;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = max_iters;
  options.function_tolerance = 1e-8;
  options.gradient_tolerance = 1e-10;
  options.parameter_tolerance = 1e-8;
  options.minimizer_progress_to_stdout = true;
  options.logging_type = ceres::PER_MINIMIZER_ITERATION;

  // Run solver
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << "  LM: " << summary.BriefReport() << "\n";

  // ---- Posterior covariance at final LM point ----
  std::cout << "  [LM covariance] theta = [";
  for (int i = 0; i < obs_cost.n_params_; ++i) std::cout << (i ? " " : "") << x[i];
  std::cout << "]\n";
  auto cov_res = compute_covariance_at(x, obs_cost, prior_cost);

  Result res;
  res.final_params = x;
  res.n_iters      = static_cast<int>(summary.iterations.size());
  res.converged    = summary.termination_type == ceres::CONVERGENCE;
  res.final_cost   = summary.final_cost;
  res.cov          = std::move(cov_res.cov);
  res.cov_sandwich = std::move(cov_res.cov_sandwich);

  return res;
}

// ---------------------------------------------------------------------------
// Covariance + sandwich computation at arbitrary theta
// ---------------------------------------------------------------------------
LMOptimizer::CovarianceResult LMOptimizer::compute_covariance_at(
    const std::vector<double>& theta,
    ObservationCost& obs_cost,
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

  for (int i = 0; i < n_obs; ++i) {
    double r   = raw_resid[i];
    double r1  = std::tanh(r);           // ρ'(r)
    double r2  = 1.0 - r1 * r1;          // ρ''(r) = sech²(r)
    Eigen::Matrix<double, 1, Eigen::Dynamic> g = J_raw.row(i);
    H += r2 * g.transpose() * g;
    V += (r1 * r1) * g.transpose() * g;
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

// ===========================================================================
// Identification pipeline
// ===========================================================================
IdentificationPipeline::Result IdentificationPipeline::run(
    const IdentificationData& data,
    const Params& base_params,
    const std::vector<ParamDescriptor>& param_desc,
    int n_cmaes_restarts,
    int cmaes_evals)
{
  int dim = static_cast<int>(param_desc.size());

  // Build cost functions
  ObservationCost obs_cost(data, base_params, param_desc);
  PriorCost prior_cost(param_desc);

  // Compute initial sigma from prior std
  std::vector<double> initial_sigma(dim);
  for (int i = 0; i < dim; ++i)
    initial_sigma[i] = param_desc[i].prior_std;

  // ---- CMA-ES restarts ----
  double best_cma_cost = 1e20;
  std::vector<double> best_cma_point(dim, 0.0);
  std::vector<RestartRecord> restart_points;
  restart_points.reserve(n_cmaes_restarts);

  for (int restart = 0; restart < n_cmaes_restarts; ++restart) {
    std::vector<double> x0(dim, 0.0);

    // Build CMA-ES parameters with random seed
    // Bounds θ ∈ [-3, 3] (log-space: multiplier ∈ [0.05, 20])
    uint64_t rng_seed = static_cast<uint64_t>(std::random_device{}());
    std::vector<double> lbounds(dim, -3.0);
    std::vector<double> ubounds(dim, 3.0);
    libcmaes::CMAParameters<> cmap(x0, initial_sigma, -1,
                                    lbounds, ubounds, rng_seed);
    cmap.set_algo(CMAES_DEFAULT);
    cmap.set_max_fevals(cmaes_evals);

    // Wrap cost and prior into CMA-ES FitFunc
    // libcmaes NoBoundStrategy silently discards bounds, so enforce them here
    libcmaes::FitFunc fit_func =
        [&obs_cost, &prior_cost, &lbounds, &ubounds](const double* x, const int& n) -> double {
      for (int i = 0; i < n; ++i)
        if (x[i] < lbounds[i] || x[i] > ubounds[i])
          return 1e20;
      double fval = obs_cost.compute_scalar_cost(x) + prior_cost.compute_scalar_prior(x);
      if (!std::isfinite(fval)) return 1e20;
      return fval;
    };

    // Run CMA-ES
    libcmaes::CMASolutions cmasols = libcmaes::cmaes(fit_func, cmap);

    auto bc = cmasols.best_candidate();
    double restart_cost = bc.get_fvalue();
    int restart_evals  = cmasols.fevals();
    const double* best_x = bc.get_x_ptr();

    if (restart_cost < best_cma_cost) {
      best_cma_cost = restart_cost;
      best_cma_point.assign(best_x, best_x + dim);
    }

    // Record every restart's best point for diagnostic gate (a)
    restart_points.push_back({std::vector<double>(best_x, best_x + dim), restart_cost});

    std::cout << "  CMA-ES restart " << (restart + 1) << "/"
              << n_cmaes_restarts << ": cost=" << restart_cost
              << " fevals=" << restart_evals << "\n";
  }

  // ---- LM refinement ---
  auto lm_res = LMOptimizer::run(best_cma_point, obs_cost, prior_cost, 100);

  // Aggregate results
  Result result;
  // If CMA-ES found nothing (default 1e20), accept LM only if it improved
  // AND produced a valid finite cost.
  bool cmaes_failed = (best_cma_cost >= 1e19);
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

  // Always compute covariance at the chosen params, regardless of branch
  std::cout << "  [pipeline covariance] theta = [";
  for (int i = 0; i < dim; ++i) std::cout << (i ? " " : "") << result.params[i];
  std::cout << "]\n";
  auto cov_res = LMOptimizer::compute_covariance_at(result.params, obs_cost, prior_cost);
  result.cov          = std::move(cov_res.cov);
  result.cov_sandwich = std::move(cov_res.cov_sandwich);

  return result;
}

Params IdentificationPipeline::Result::to_params(
    const Params& base,
    const std::vector<ParamDescriptor>& desc) const
{
  Params p = base;
  for (size_t i = 0; i < params.size() && i < desc.size(); ++i) {
    double mult = std::exp(params[i]);

    if (desc[i].name == "kp1")  p.kp1  *= mult;
    if (desc[i].name == "kp2")  p.kp2  *= mult;
    if (desc[i].name == "kp3")  p.kp3  *= mult;
    if (desc[i].name == "Vm0")  p.Vm0  *= mult;
    if (desc[i].name == "Vmx")  p.Vmx  *= mult;
    if (desc[i].name == "p2U")  p.p2U  *= mult;
    if (desc[i].name == "ka1")  p.ka1  *= mult;
    if (desc[i].name == "ka2")  p.ka2  *= mult;
    if (desc[i].name == "kd")   p.kd   *= mult;
    if (desc[i].name == "kmax") p.kmax *= mult;
    if (desc[i].name == "kmin") p.kmin *= mult;
    if (desc[i].name == "kabs") p.kabs *= mult;
    if (desc[i].name == "kgri") p.kgri *= mult;
    if (desc[i].name == "VG")   p.VG   *= mult;
    if (desc[i].name == "VI")   p.VI   *= mult;
    if (desc[i].name == "Gb")   p.Gb   *= mult;
    if (desc[i].name == "Ib")   p.Ib   *= mult;
    if (desc[i].name == "k1")   p.k1   *= mult;
    if (desc[i].name == "k2")   p.k2   *= mult;
    if (desc[i].name == "m1")   p.m1   *= mult;
    if (desc[i].name == "m2")   p.m2   *= mult;
    if (desc[i].name == "m4")   p.m4   *= mult;
    if (desc[i].name == "ki")   p.ki   *= mult;
    if (desc[i].name == "f")    p.f    *= mult;
    if (desc[i].name == "b")    p.b    *= mult;
    if (desc[i].name == "d")    p.d    *= mult;
    if (desc[i].name == "ke1")  p.ke1  *= mult;
    if (desc[i].name == "ke2")  p.ke2  *= mult;
    if (desc[i].name == "Ts")   p.Ts   *= mult;
    if (desc[i].name == "r1")   p.r1   *= mult;
    if (desc[i].name == "Gth")  p.Gth  *= mult;
    if (desc[i].name == "BW")   p.BW   *= mult;
  }
  return p;
}

// ===========================================================================
// Pipeline convenience: load from JSON config
// ===========================================================================
IdentificationPipeline::Result IdentificationPipeline::run_from_config(
    const IdentificationData& data,
    const Params& base_params,
    const std::string& json_path,
    int n_cmaes_restarts,
    int cmaes_evals,
    std::vector<ParamDescriptor>* full_desc)
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

} // namespace uva_padova
