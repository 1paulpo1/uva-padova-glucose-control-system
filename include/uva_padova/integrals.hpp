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

    // Physiological clamping (hard for double, smooth for Jet)
    x.clamp_physiological();
  }

  // Apply a discrete meal impulse (always double — dosing decision)
  static void apply_meal_event(BasicState<double>& state, double cho_mg);
};

// ---------------------------------------------------------------------------
// RK5 integrator (Dormand-Prince 5(4)) for the UVA-Padova ODE.
// 5th order with 6 function evaluations per step.
// Templated on scalar type T to support autodiff (ceres::Jet).
// ---------------------------------------------------------------------------
struct RK5Integrator {
  // Advance the state by dt minutes using a single RK5 (DP5) step.
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

    T h(dt);

    // Stage k1
    BasicState<T> k1 = UvaPadovaODESystem::rhs(x, u, p);

    // Stage k2
    BasicState<T> x2 = add(x, k1, h * T(1.0/5.0));
    BasicState<T> k2 = UvaPadovaODESystem::rhs(x2, u, p);

    // Stage k3
    BasicState<T> x3 = add(add(x, k1, h * T(3.0/40.0)), k2, h * T(9.0/40.0));
    BasicState<T> k3 = UvaPadovaODESystem::rhs(x3, u, p);

    // Stage k4
    BasicState<T> x4 = add(add(add(x, k1, h * T(44.0/45.0)),
                               k2, h * T(-56.0/15.0)),
                           k3, h * T(32.0/9.0));
    BasicState<T> k4 = UvaPadovaODESystem::rhs(x4, u, p);

    // Stage k5
    BasicState<T> x5 = add(add(add(add(x, k1, h * T(19372.0/6561.0)),
                                   k2, h * T(-25360.0/2187.0)),
                               k3, h * T(64448.0/6561.0)),
                           k4, h * T(-212.0/729.0));
    BasicState<T> k5 = UvaPadovaODESystem::rhs(x5, u, p);

    // Stage k6
    BasicState<T> x6 = add(add(add(add(add(x, k1, h * T(9017.0/3168.0)),
                                       k2, h * T(-355.0/33.0)),
                                   k3, h * T(46732.0/5247.0)),
                               k4, h * T(49.0/176.0)),
                           k5, h * T(-5103.0/18656.0));
    BasicState<T> k6 = UvaPadovaODESystem::rhs(x6, u, p);

    // 5th order update: b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    for (int i = 0; i < BasicState<T>::kSize; ++i) {
      x.at(i) += h * (T(35.0/384.0) * k1.at(i)
                    + T(500.0/1113.0) * k3.at(i)
                    + T(125.0/192.0) * k4.at(i)
                    - T(2187.0/6784.0) * k5.at(i)
                    + T(11.0/84.0) * k6.at(i));
    }

    // Safety clamp — only active for double
    x.clamp_physiological();
  }
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
