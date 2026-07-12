#pragma once

#include <functional>
#include <vector>
#include <utility>

#include "uva_padova/types.hpp"
#include "uva_padova/cost_function.hpp"

namespace uva_padova {

// Mock simulator for unit-testing ObservationCost.
//
// Behaviour is configured via static members (set before each test):
//   ss_state       — steady state returned by find_steady_state
//   basal_insulin  — basal rate returned by find_steady_state
//   step_delta     — optional callback called each step (double only)
struct MockSimulator {
  static inline State ss_state{};
  static inline double basal_insulin{0.0};
  static inline int step_count{0};
  static inline double (*step_delta)(const BasicParams<double>&) = nullptr;
  static inline std::function<void(const BasicState<double>&,
                                    const BasicInputs<double>&)> step_observer;
  static inline std::vector<std::pair<int,int>> event_log;

  static void reset() {
    ss_state = State{};
    basal_insulin = 0.0;
    step_count = 0;
    step_delta = nullptr;
    step_observer = std::function<void(const BasicState<double>&,
                                        const BasicInputs<double>&)>{};
    event_log.clear();
  }

  template<int EventType, typename T>
  static void notify(const BasicState<T>&, const BasicInputs<T>&, T t) {
    if constexpr (std::is_same_v<T, double>) {
      event_log.push_back({EventType, static_cast<int>(t)});
    }
  }

  static double find_steady_state(State& out,
                                  const BasicParams<double>& /*p*/,
                                  double /*Gb*/) {
    out = ss_state;
    return basal_insulin;
  }

  template<typename T>
  static void step(BasicState<T>& state, const BasicInputs<T>& u,
                   const BasicParams<T>& p, double /*dt*/) {
    if constexpr (std::is_same_v<T, double>) {
      ++step_count;
      if (step_observer) step_observer(state, u);
      if (step_delta) state.Gsc += T(step_delta(p));
    }
  }
};

// Simulator with analytically-differentiable linear dynamics:
//   Gsc += dt * p.kp1   each step
// Steady state is fixed (parameter-independent).
// Used for gradient correctness tests.
struct LinearMockSimulator {
  static inline State ss_state{};
  static inline double basal_insulin{0.0};

  static void reset() {
    ss_state = State{};
    basal_insulin = 0.0;
  }

  static double find_steady_state(State& out,
                                  const BasicParams<double>& /*p*/,
                                  double /*Gb*/) {
    out = ss_state;
    return basal_insulin;
  }

  template<typename T>
  static void step(BasicState<T>& state, const BasicInputs<T>& /*u*/,
                   const BasicParams<T>& p, double dt) {
    state.Gsc += T(dt) * p.kp1;
  }

  template<int EventType, typename T>
  static void notify(const BasicState<T>&, const BasicInputs<T>&, T) {}
};

} // namespace uva_padova
