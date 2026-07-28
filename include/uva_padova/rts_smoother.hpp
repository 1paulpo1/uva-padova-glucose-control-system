#pragma once

#include <vector>
#include <functional>
#include <Eigen/Dense>

namespace uva_padova {

// ===========================================================================
//  RTS Smoother (Rauch–Tung–Striebel)
//
//  Given forward-filtered state estimates and covariances, computes
//  smoothed (retrospective) estimates with lower variance.
// ===========================================================================

struct RTSSmoother {
  struct FilteredStep {
    double t;
    Eigen::VectorXd x_pred;
    Eigen::MatrixXd P_pred;
    Eigen::VectorXd x_filt;
    Eigen::MatrixXd P_filt;
  };

  struct Result {
    std::vector<Eigen::VectorXd> x_smooth;
    std::vector<Eigen::MatrixXd> P_smooth;
    std::vector<Eigen::MatrixXd> G;
  };

  // Jacobian callback: F_k = ∂f/∂x evaluated at x̂_{k|k}, t_k
  std::function<Eigen::MatrixXd(const Eigen::VectorXd&, double)>
    compute_jacobian;

  RTSSmoother() = default;

  explicit RTSSmoother(
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&, double)>&& jac)
    : compute_jacobian(std::move(jac))
  {}

  Result smooth(const std::vector<FilteredStep>& steps) const {
    int N = static_cast<int>(steps.size());
    if (N < 2) return {};

    Result result;
    result.x_smooth.resize(N);
    result.P_smooth.resize(N);
    result.G.resize(N - 1);

    result.x_smooth[N - 1] = steps[N - 1].x_filt;
    result.P_smooth[N - 1] = steps[N - 1].P_filt;

    for (int k = N - 2; k >= 0; --k) {
      const auto& step = steps[k];
      const auto& step_next = steps[k + 1];
      const Eigen::VectorXd& x_pred_next = step_next.x_pred;

      int n = static_cast<int>(step.x_filt.size());

      Eigen::MatrixXd F;
      if (compute_jacobian) {
        F = compute_jacobian(step.x_filt, step.t);
      } else {
        F = Eigen::MatrixXd::Identity(n, n);
      }

      Eigen::MatrixXd G_k = step.P_filt * F.transpose() *
                            step_next.P_pred.llt().solve(Eigen::MatrixXd::Identity(n, n));

      result.x_smooth[k] = step.x_filt +
                           G_k * (result.x_smooth[k + 1] - x_pred_next);
      result.P_smooth[k] = step.P_filt +
                           G_k * (result.P_smooth[k + 1] - step_next.P_pred) * G_k.transpose();
      result.G[k] = G_k;
    }

    return result;
  }
};

} // namespace uva_padova
