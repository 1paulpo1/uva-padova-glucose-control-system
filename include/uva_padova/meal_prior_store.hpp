#pragma once

#include <vector>
#include <cmath>

namespace uva_padova {

// ===========================================================================
//  Meal-prior store
//
//  Context-indexed Bayesian averaging with exponential forgetting.
//  Stores RTS-smoothed parameter estimates for meal absorption params
//  and re-uses them as initial conditions for the next similar meal.
// ===========================================================================

struct MealPriorStore {
  enum class MealType { BREAKFAST, LUNCH, DINNER, SNACK };

  struct Prior {
    double hour_of_day;
    std::vector<double> theta_meal;
    double lambda;
    int n_observations;
  };

  std::vector<Prior> priors;
  double lambda_new{0.95};
  double lambda_stable{0.75};

  static MealType classify_meal_type(double hour) {
    if (hour >= 5 && hour < 10) return MealType::BREAKFAST;
    if (hour >= 11 && hour < 14) return MealType::LUNCH;
    if (hour >= 17 && hour < 21) return MealType::DINNER;
    return MealType::SNACK;
  }

  double hour_of_day(double t_min) const {
    return std::fmod(t_min, 24.0 * 60.0) / 60.0;
  }

  bool get_prior(double t_min, std::vector<double>& theta_out) const {
    double h = hour_of_day(t_min);
    MealType type = classify_meal_type(h);

    for (const auto& p : priors) {
      MealType pt = classify_meal_type(p.hour_of_day);
      if (pt == type && p.n_observations > 0) {
        theta_out = p.theta_meal;
        return true;
      }
    }
    return false;
  }

  void update_prior(double t_min,
                    const std::vector<double>& theta_rts,
                    const std::vector<bool>& is_meal_param)
  {
    double h = hour_of_day(t_min);
    MealType type = classify_meal_type(h);

    std::vector<double> theta_meal;
    for (size_t i = 0; i < is_meal_param.size(); ++i) {
      if (is_meal_param[i] && i < theta_rts.size())
        theta_meal.push_back(theta_rts[i]);
    }

    if (theta_meal.empty()) return;

    auto it = priors.begin();
    while (it != priors.end()) {
      if (classify_meal_type(it->hour_of_day) == type)
        break;
      ++it;
    }

    if (it == priors.end()) {
      double lambda = lambda_new;
      priors.push_back({h, theta_meal, lambda, 1});
    } else {
      it->n_observations++;
      double lambda = (it->n_observations < 3) ? lambda_new :
                      (it->n_observations >= 10) ? lambda_stable :
                      0.85;

      for (size_t i = 0; i < theta_meal.size(); ++i) {
        if (i < it->theta_meal.size())
          it->theta_meal[i] = lambda * it->theta_meal[i] + (1.0 - lambda) * theta_meal[i];
      }
      it->lambda = lambda;
    }
  }

  int size() const { return static_cast<int>(priors.size()); }
};

} // namespace uva_padova
