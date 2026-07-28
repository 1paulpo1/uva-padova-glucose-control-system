#pragma once

#include <vector>
#include <Eigen/Dense>

#include "uva_padova/types.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/integrals.hpp"
#include "uva_padova/simulator.hpp"
#include "uva_padova/cost_function.hpp"
#include "uva_padova/augmented_state.hpp"

namespace uva_padova {

// ===========================================================================
//  System model: ODE propagation for model states + random walk for θ
//
//  Used as the f(·) operator inside CubatureKalmanFilter::predict().
//  Propagates the augmented state forward by dt via RK4 on the UVA-Padova
//  ODE, adjusting model parameters with the current θ multipliers.
// ===========================================================================

struct SystemModel {
  const BasicParams<double> base_params;
  const std::vector<ParamDescriptor> param_desc;
  const double dt;         // integration time step [min]
  const double sub_dt;     // sub-step for RK4 (dt / n_sub)
  const int n_sub;

  SystemModel(const BasicParams<double>& params,
              const std::vector<ParamDescriptor>& desc,
              double step_min,
              int n_sub_steps = 5)
    : base_params(params),
      param_desc(desc),
      dt(step_min),
      sub_dt(step_min / static_cast<double>(n_sub_steps)),
      n_sub(n_sub_steps)
  {}

  BasicParams<double> adjusted_params(const double* theta) const {
    BasicParams<double> p = base_params;
    for (size_t i = 0; i < param_desc.size(); ++i) {
      double mult = std::exp(theta[i]);
      if (!std::isfinite(mult)) mult = 1.0;
      param_by_id(p, param_id_from_string(param_desc[i].name)) *= mult;
    }
    return p;
  }

  // operator(): advance augmented state by dt
  void operator()(Eigen::VectorXd& x_aug) const {
    const int n_theta = AugmentedStateMap<double>::n_theta(x_aug);

    BasicState<double> s = AugmentedStateMap<double>::model_state(x_aug);

    std::vector<double> theta(n_theta);
    for (int i = 0; i < n_theta; ++i)
      theta[i] = AugmentedStateMap<double>::theta(x_aug, i);

    BasicParams<double> p = adjusted_params(theta.data());

    double basal_insulin = 0.0;
    BasicParams<double> p_double = p;
    {
      State ss_state;
      double insulin_ss = UvaPadovaSimulator::find_steady_state(ss_state, p_double, p_double.Gb);
      if (std::isfinite(insulin_ss) && insulin_ss >= 0) {
        basal_insulin = insulin_ss;
      }
    }

    BasicInputs<double> u;
    u.insulin_rate = basal_insulin;

    for (int k = 0; k < n_sub; ++k) {
      RK4Integrator::step(UvaPadovaODESystem{}, s, u, p, sub_dt);
      s.clamp_physiological();
    }

    AugmentedStateMap<double>::set_model_state(x_aug, s);

    // Theta remains unchanged (random walk: θ_{k} = θ_{k-1})
    // Q handles the diffusion
  }
};

} // namespace uva_padova
