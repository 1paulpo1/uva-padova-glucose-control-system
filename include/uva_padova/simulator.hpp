#pragma once

#include "types.hpp"
#include "odesystem.hpp"
#include "integrals.hpp"

namespace uva_padova {

// ---------------------------------------------------------------------------
// UVA-Padova simulator — concrete policy for production use.
//
// Satisfies the concept:
//   static double find_steady_state(State& ss_state,
//                                   const BasicParams<double>& p,
//                                   double Gb);
//   template<typename T>
//   static void step(BasicState<T>& state,
//                    const BasicInputs<T>& u,
//                    const BasicParams<T>& p,
//                    double dt);
// ---------------------------------------------------------------------------
struct UvaPadovaSimulator {
  static double find_steady_state(State& ss_state,
                                  const BasicParams<double>& p,
                                  double Gb) {
    return SteadyStateSolver::find(ss_state, p, Gb);
  }

  template<typename T>
  static void step(BasicState<T>& state,
                   const BasicInputs<T>& u,
                   const BasicParams<T>& p,
                   double dt) {
    RK4Integrator::step(UvaPadovaODESystem{}, state, u, p, dt);
  }
};

} // namespace uva_padova
