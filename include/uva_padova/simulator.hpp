#pragma once

#include "types.hpp"
#include "odesystem.hpp"
#include "integrals.hpp"

namespace uva_padova {

// ---------------------------------------------------------------------------
// UVA-Padova simulator — concrete policy for production use.
//
// Satisfies the Simulator concept (implemented by UvaPadovaSimulator,
// MockSimulator, LinearMockSimulator):
//
//   // Return steady-state glucose and basal insulin rate.
//   static double find_steady_state(State& ss_state,
//                                   const BasicParams<double>& p,
//                                   double Gb);
//
//   // Advance state by dt minutes.
//   template<typename T>
//   static void step(BasicState<T>& state,
//                    const BasicInputs<T>& u,
//                    const BasicParams<T>& p,
//                    double dt);
//
//   // Event notification — called after each event block (insulin/bolus/meal).
//   // EventType: 0=insulin, 1=bolus, 2=meal.
//   template<int EventType, typename T>
//   static void notify(const BasicState<T>& state,
//                      const BasicInputs<T>& u,
//                      T t);
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

  template<int EventType, typename T>
  static void notify(const BasicState<T>&, const BasicInputs<T>&, T) {}
};

} // namespace uva_padova
