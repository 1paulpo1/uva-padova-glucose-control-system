#pragma once

#include <Eigen/Dense>

#include "uva_padova/types.hpp"

namespace uva_padova {

// ===========================================================================
//  Augmented state: [14 model states | θ₀ ... θ_{n-1}]  (CKF convention)
//  Indices:  0–13  model states (BasicState ordering)
//            14+   adaptive parameters (log-space θ, same as offline)
// ===========================================================================

// Map between flat vector and structured state
template<typename T>
struct AugmentedStateMap {
  static constexpr int kModelDim = BasicState<T>::kSize; // 14

  static BasicState<T> model_state(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) {
    BasicState<T> s;
    for (int i = 0; i < kModelDim; ++i)
      s.at(i) = x(i);
    return s;
  }

  static void set_model_state(Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
                               const BasicState<T>& s) {
    for (int i = 0; i < kModelDim; ++i)
      x(i) = s.at(i);
  }

  static T theta(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x, int idx) {
    return x(kModelDim + idx);
  }

  static void set_theta(Eigen::Matrix<T, Eigen::Dynamic, 1>& x, int idx, T val) {
    x(kModelDim + idx) = val;
  }

  static int n_theta(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) {
    return static_cast<int>(x.size()) - kModelDim;
  }
};

} // namespace uva_padova
