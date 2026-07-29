// IMPORTANT: include libcmaes BEFORE Ceres/glog to avoid LOG macro conflict
#include <libcmaes/cmaes.h>

#include "uva_padova/optimizer.hpp"
#include "uva_padova/config_loader.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

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
  return run(std::move(fit_func),
             initial_sigma,
             max_evals,
             seed,
             std::vector<double>(dim, lower_bound),
             std::vector<double>(dim, upper_bound));
}

CMAESOptimizer::Result CMAESOptimizer::run(
    std::function<double(const double*, int)> fit_func,
    const std::vector<double>& initial_sigma,
    int max_evals,
    int seed,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds)
{
  int dim = static_cast<int>(initial_sigma.size());
  if (static_cast<int>(lower_bounds.size()) != dim ||
      static_cast<int>(upper_bounds.size()) != dim) {
    throw std::invalid_argument("CMAESOptimizer::run: bounds size mismatch");
  }

  std::vector<double> sigma_vec = initial_sigma;

  if (seed < 0)
    seed = static_cast<int>(std::random_device{}());

  std::vector<double> x0(dim, 0.0);
  std::vector<double> lbounds = lower_bounds;
  std::vector<double> ubounds = upper_bounds;

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

// LMOptimizer::run, compute_covariance, and compute_covariance_at are
// templated on Simulator — definitions are in optimizer_impl.hpp
// (included from optimizer.hpp).
} // namespace uva_padova
