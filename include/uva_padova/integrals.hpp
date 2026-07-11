#pragma once

#include "types.hpp"
#include "odesystem.hpp"

namespace uva_padova {

// ---------------------------------------------------------------------------
// RK4 integrator for the UVA-Padova ODE.
// Templated on scalar type T to support autodiff (ceres::Jet).
// ---------------------------------------------------------------------------
struct RK4Integrator {
  // Advance the state by dt minutes using a single RK4 step.
  template<typename T>
  static void step(const UvaPadovaODESystem& /*sys*/,
                   BasicState<T>& x,
                   const BasicInputs<T>& u,
                   const BasicParams<T>& p,
                   double dt) {
    auto add = [](const BasicState<T>& a, const BasicState<T>& b,
                  T scale) -> BasicState<T> {
      BasicState<T> r;
      for (int i = 0; i < BasicState<T>::kSize; ++i)
        r.at(i) = a.at(i) + scale * b.at(i);
      return r;
    };

    // k1
    BasicState<T> k1 = UvaPadovaODESystem::rhs(x, u, p);

    // k2
    BasicState<T> x2 = add(x, k1, T(0.5 * dt));
    BasicState<T> k2 = UvaPadovaODESystem::rhs(x2, u, p);

    // k3
    BasicState<T> x3 = add(x, k2, T(0.5 * dt));
    BasicState<T> k3 = UvaPadovaODESystem::rhs(x3, u, p);

    // k4
    BasicState<T> x4 = add(x, k3, T(dt));
    BasicState<T> k4 = UvaPadovaODESystem::rhs(x4, u, p);

    // weighted sum
    for (int i = 0; i < BasicState<T>::kSize; ++i) {
      x.at(i) += T(dt / 6.0) * (k1.at(i) + T(2.0) * k2.at(i)
                                 + T(2.0) * k3.at(i) + k4.at(i));
    }

    // Safety clamp — only active for double
    x.clamp_physiological();
  }

  // Apply a discrete meal impulse (always double — dosing decision)
  static void apply_meal_event(BasicState<double>& state, double cho_mg);
};

// ---------------------------------------------------------------------------
// Steady-state solver (double only).
// ---------------------------------------------------------------------------
struct SteadyStateSolver {
  static double find(BasicState<double>& state_out, const Params& p,
                     double target_Gb = -1.0);
  static double forward_warmup(BasicState<double>& state, const Params& p,
                                double insulin_rate, int duration_min);
};

} // namespace uva_padova
