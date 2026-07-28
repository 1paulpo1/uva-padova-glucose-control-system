#pragma once

#include <cmath>
#include <Eigen/Dense>

namespace uva_padova {

// ===========================================================================
//  Cubature Kalman Filter (CKF)
//
//  CKF = UKF with α=1, β=0, κ=0 → spherical-radial cubature rule.
//  All 2n cubature points have equal weight w_i = 1/(2n).
//
//  Generic dynamic-size implementation that works with any state dimension.
// ===========================================================================
class CubatureKalmanFilter {
public:
  // ---- Construction ----
  CubatureKalmanFilter(int n_theta,
                        const Eigen::VectorXd& x0,
                        const Eigen::MatrixXd& P0,
                        const Eigen::MatrixXd& Q,
                        const Eigen::MatrixXd& R)
    : n_theta_(n_theta),
      dim_(x0.size()),
      x_(x0),
      P_(P0),
      Q_(Q),
      R_(R)
  {}

  // ---- Accessors ----
  int dim() const { return dim_; }
  int n_theta() const { return n_theta_; }
  const Eigen::VectorXd& state() const { return x_; }
  const Eigen::MatrixXd& covariance() const { return P_; }
  const Eigen::MatrixXd& process_noise() const { return Q_; }
  const Eigen::MatrixXd& measurement_noise() const { return R_; }

  // ---- Modify Q at runtime (regime-dependent modulation) ----
  void set_process_noise(const Eigen::MatrixXd& Q) {
    Q_ = Q;
  }

  // Apply a caller-defined mutation to state and covariance.
  template<typename F>
  void modify_state(F&& f) {
    f(x_, P_);
  }

  // ---- Predict step ----
  // Propagates cubature points through the system function f().
  // f must be callable as: void f(Eigen::VectorXd& x)
  template<typename F>
  void predict(F&& f) {
    const int n = dim_;
    const int n_pts = 2 * n;

    // 1. Cholesky of current covariance
    Eigen::MatrixXd S = P_.llt().matrixL();

    // 2. Generate cubature points: ξ_i = x̂ + √n · [S]_i
    // 3. Propagate through system model (copy to temp for Ref-compatibility)
    Eigen::MatrixXd Xi_pred(n, n_pts);
    double sqrt_n = std::sqrt(static_cast<double>(n));
    for (int i = 0; i < n; ++i) {
      Eigen::VectorXd xi = x_ + sqrt_n * S.col(i);
      f(xi);
      Xi_pred.col(i) = xi;

      xi = x_ - sqrt_n * S.col(i);
      f(xi);
      Xi_pred.col(i + n) = xi;
    }

    // 4. Predicted mean: weighted average (all weights = 1/(2n))
    double w = 1.0 / static_cast<double>(n_pts);
    x_.setZero();
    for (int i = 0; i < n_pts; ++i)
      x_ += w * Xi_pred.col(i);

    // 5. Predicted covariance
    P_.setZero();
    for (int i = 0; i < n_pts; ++i) {
      Eigen::VectorXd d = Xi_pred.col(i) - x_;
      P_ += w * d * d.transpose();
    }
    P_ += Q_;
  }

  // ---- Update step ----
  // Corrects state estimate using measurement z.
  // h must be callable as: Eigen::VectorXd h(const Eigen::VectorXd& x)
  // Returns innovation (z - y_pred) for diagnostics.
  template<typename H>
  Eigen::VectorXd update(const Eigen::VectorXd& z, H&& h) {
    const int n = dim_;
    const int m = static_cast<int>(z.size());
    const int n_pts = 2 * n;

    // 1. Generate cubature points from predicted x_, P_
    Eigen::MatrixXd S = P_.llt().matrixL();
    double sqrt_n = std::sqrt(static_cast<double>(n));
    Eigen::MatrixXd Xi(n, n_pts);
    for (int i = 0; i < n; ++i) {
      Xi.col(i)     = x_ + sqrt_n * S.col(i);
      Xi.col(i + n) = x_ - sqrt_n * S.col(i);
    }

    // 2. Propagate through measurement model
    Eigen::MatrixXd Psi(m, n_pts);
    for (int i = 0; i < n_pts; ++i)
      Psi.col(i) = h(Xi.col(i));

    // 3. Predicted measurement: mean
    double w = 1.0 / static_cast<double>(n_pts);
    Eigen::VectorXd y_pred(m);
    y_pred.setZero();
    for (int i = 0; i < n_pts; ++i)
      y_pred += w * Psi.col(i);

    // 4. Innovation covariance P_yy
    Eigen::MatrixXd P_yy(m, m);
    P_yy.setZero();
    for (int i = 0; i < n_pts; ++i) {
      Eigen::VectorXd d = Psi.col(i) - y_pred;
      P_yy += w * d * d.transpose();
    }
    P_yy += R_;

    // 5. Cross-covariance P_xy
    Eigen::MatrixXd P_xy(n, m);
    P_xy.setZero();
    for (int i = 0; i < n_pts; ++i) {
      Eigen::VectorXd dx = Xi.col(i) - x_;
      Eigen::VectorXd dy = Psi.col(i) - y_pred;
      P_xy += w * dx * dy.transpose();
    }

    // 6. Kalman gain K = P_xy * P_yy^{-1}
    Eigen::MatrixXd K = P_xy * P_yy.llt().solve(Eigen::MatrixXd::Identity(m, m));

    // 7. Update
    Eigen::VectorXd innov = z - y_pred;
    x_ += K * innov;
    P_ -= K * P_yy * K.transpose();

    return innov;
  }

  // ---- Reset ----
  void reset(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) {
    x_ = x;
    P_ = P;
  }

private:
  int n_theta_;
  int dim_;
  Eigen::VectorXd x_;       // state estimate
  Eigen::MatrixXd P_;       // state covariance
  Eigen::MatrixXd Q_;       // process noise covariance
  Eigen::MatrixXd R_;       // measurement noise covariance
};

} // namespace uva_padova
