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

// ===========================================================================
// Ceres Levenberg-Marquardt optimizer
// ===========================================================================
LMOptimizer::Result LMOptimizer::run(
    const std::vector<double>& initial_params,
    ObservationCost<>& obs_cost,
    PriorCost& prior_cost,
    int max_iters)
{
  std::vector<double> x = initial_params;

  // Build Ceres problem
  ceres::Problem::Options problem_options;
  problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  // Observation residuals — plain (G-obs)/σ, robustness via SoftLOneLoss.
  problem.AddResidualBlock(&obs_cost, new ceres::SoftLOneLoss(1.0), x.data());

  // MAP prior residuals (quadratic, no loss function)
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

  // ---- Posterior covariance via Ceres (auto-mode, loss-weighted) ----
  ceres::Covariance::Options cov_options;
  cov_options.apply_loss_function = true;
  ceres::Covariance covariance(cov_options);

  Eigen::MatrixXd cov_matrix;
  {
    int dim = obs_cost.n_params_;
    std::vector<std::pair<const double*, const double*>> cov_blocks;
    cov_blocks.push_back({x.data(), x.data()});
    if (covariance.Compute(cov_blocks, &problem)) {
      cov_matrix.resize(dim, dim);
      covariance.GetCovarianceBlock(x.data(), x.data(), cov_matrix.data());
    } else {
      cov_matrix = Eigen::MatrixXd::Identity(dim, dim) * 1e12;
    }
  }

  std::cout << "  LM: " << summary.BriefReport() << "\n";

  Result res;
  res.final_params = x;
  res.n_iters      = static_cast<int>(summary.iterations.size());
  res.converged    = summary.termination_type == ceres::CONVERGENCE;
  res.final_cost   = summary.final_cost;
  res.cov          = std::move(cov_matrix);
  // cov_sandwich retained for reference; unused in auto-mode

  return res;
}

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

// ---------------------------------------------------------------------------
// Ceres Covariance block at arbitrary theta (extracted from Pipeline fallback)
// ---------------------------------------------------------------------------
LMOptimizer::CovarianceResult LMOptimizer::compute_covariance(
    const std::vector<double>& theta,
    ObservationCost<>& obs_cost,
    PriorCost& prior_cost)
{
  int dim = static_cast<int>(theta.size());
  if (dim == 0) return {};

  std::vector<double> x = theta;

  ceres::Problem tmp_problem;
  tmp_problem.AddResidualBlock(&obs_cost, new ceres::SoftLOneLoss(1.0), x.data());
  tmp_problem.AddResidualBlock(&prior_cost, nullptr, x.data());

  ceres::Covariance::Options cov_opts;
  ceres::Covariance tmp_cov(cov_opts);
  std::vector<std::pair<const double*, const double*>> cb = {
      {x.data(), x.data()}};

  Eigen::MatrixXd cov_mat = Eigen::MatrixXd::Identity(dim, dim) * 1e12;
  if (tmp_cov.Compute(cb, &tmp_problem)) {
    cov_mat.resize(dim, dim);
    tmp_cov.GetCovarianceBlock(x.data(), x.data(), cov_mat.data());
  }

  return {cov_mat, {}};
}

// ===========================================================================
// Identification pipeline
// ===========================================================================
IdentificationPipeline::Result IdentificationPipeline::run(
    const IdentificationData& data,
    const Params& base_params,
    const std::vector<ParamDescriptor>& param_desc,
    int n_cmaes_restarts,
    int cmaes_evals,
    int lm_max_iters,
    double cmaes_lower,
    double cmaes_upper)
{
  int dim = static_cast<int>(param_desc.size());

  // Build cost functions
  ObservationCost<> obs_cost(data, base_params, param_desc);
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

  // Build cost + prior into CMA-ES fit function (lambda shared across restarts)
  std::function<double(const double*, int)> fit_func =
      [&obs_cost, &prior_cost](const double* x, int n) -> double {
    (void)n;
    double fval = obs_cost.compute_scalar_cost(x) + prior_cost.compute_scalar_prior(x);
    if (!std::isfinite(fval)) return 1e20;
    return fval;
  };

  for (int restart = 0; restart < n_cmaes_restarts; ++restart) {
    int seed = static_cast<int>(std::random_device{}());
    auto cmaes_res = CMAESOptimizer::run(fit_func, initial_sigma, cmaes_evals,
                                          seed, cmaes_lower, cmaes_upper);

    if (cmaes_res.best_cost < best_cma_cost) {
      best_cma_cost = cmaes_res.best_cost;
      best_cma_point = cmaes_res.best_params;
    }

    restart_points.push_back({cmaes_res.best_params, cmaes_res.best_cost});

    std::cout << "  CMA-ES restart " << (restart + 1) << "/"
              << n_cmaes_restarts << ": cost=" << cmaes_res.best_cost
              << " fevals=" << cmaes_res.n_evals << "\n";
  }

  // ---- LM refinement ---
  auto lm_res = LMOptimizer::run(best_cma_point, obs_cost, prior_cost, lm_max_iters);

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

  // Covariance at chosen params — Ceres auto-mode (loss-weighted)
  if (result.success && !result.params.empty()) {
    int dim = obs_cost.n_params_;
    if (lm_res.cov.rows() == dim && result.params == lm_res.final_params) {
      result.cov = lm_res.cov;
    } else {
      auto cov_res = LMOptimizer::compute_covariance(result.params, obs_cost, prior_cost);
      result.cov = cov_res.cov;
    }
    std::cout << "  [debug] covariance done\n";
  }
  result.cov_sandwich = Eigen::MatrixXd{};  // unused in auto-mode

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
