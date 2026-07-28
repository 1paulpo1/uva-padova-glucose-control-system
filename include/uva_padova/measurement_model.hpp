#pragma once

#include <Eigen/Dense>

namespace uva_padova {

// ===========================================================================
//  Measurement model: extract Gsc from augmented state
// ===========================================================================

struct MeasurementModel {
  int n_theta;

  explicit MeasurementModel(int n) : n_theta(n) {}

  Eigen::VectorXd operator()(const Eigen::VectorXd& x_aug) const {
    Eigen::VectorXd z(1);
    z(0) = x_aug(12); // Gsc at index 12 in BasicState order
    return z;
  }

  int measurement_dim() const { return 1; }
};

} // namespace uva_padova
