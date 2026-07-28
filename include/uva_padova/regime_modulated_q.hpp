#pragma once

#include <vector>
#include <Eigen/Dense>

#include "uva_padova/cost_function.hpp"

namespace uva_padova {

// ===========================================================================
//  Regime-dependent Q modulation
//
//  For each theta associated with meal absorption (e.g., kabs, kmax, kmin,
//  kgri, f), Q_ii is set to nominal only within the postprandial window.
//  Outside the window, Q_ii is frozen (multiplied by β ≪ 1).
// ===========================================================================

struct RegimeModulatedQ {
  static constexpr const char* meal_params[] = {
    "kabs", "kmax", "kmin", "kgri", "f", nullptr
  };

  std::vector<double> active_meal_end_times;
  double window_minutes{180.0};
  double freeze_factor{0.001};

  int n_theta_;
  std::vector<bool> is_meal_param_;
  Eigen::MatrixXd Q_nominal_;
  Eigen::VectorXd obs_scores_;

  RegimeModulatedQ(int n_theta, const std::vector<ParamDescriptor>& desc)
    : n_theta_(n_theta)
  {
    is_meal_param_.resize(n_theta_, false);
    for (int i = 0; i < n_theta_; ++i) {
      for (int mp = 0; meal_params[mp] != nullptr; ++mp) {
        if (desc[i].name == meal_params[mp]) {
          is_meal_param_[i] = true;
          break;
        }
      }
    }
  }

  void set_observability_scores(const Eigen::VectorXd& s) {
    obs_scores_ = s;
  }

  Eigen::MatrixXd compute(double t_min) const {
    bool in_window = false;
    for (double end : active_meal_end_times) {
      double start = end - window_minutes;
      if (t_min >= start && t_min <= end) {
        in_window = true;
        break;
      }
    }

    Eigen::MatrixXd Q = Q_nominal_;

    bool have_scores = obs_scores_.size() == Q_nominal_.rows();

    for (int i = 0; i < n_theta_; ++i) {
      int idx = 14 + i;

      if (!have_scores) {
        if (is_meal_param_[i] && !in_window)
          Q(idx, idx) *= freeze_factor;
        continue;
      }

      double obs_clip = std::min(1.0, std::abs(obs_scores_(idx)));
      double factor = freeze_factor + (1.0 - freeze_factor) * obs_clip;
      Q(idx, idx) *= factor;
    }

    return Q;
  }

  void register_meal(double meal_time_min) {
    active_meal_end_times.push_back(meal_time_min + window_minutes);
    while (active_meal_end_times.size() > 10)
      active_meal_end_times.erase(active_meal_end_times.begin());
  }

  void prune_completed(double current_time_min) {
    auto it = active_meal_end_times.begin();
    while (it != active_meal_end_times.end()) {
      if (*it < current_time_min - window_minutes)
        it = active_meal_end_times.erase(it);
      else
        ++it;
    }
  }
};

} // namespace uva_padova
