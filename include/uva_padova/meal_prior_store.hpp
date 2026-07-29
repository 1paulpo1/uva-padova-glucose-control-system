#pragma once

#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

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
    MealType type;
    double hour_of_day;
    std::vector<double> theta_meal;
    double lambda;
    int n_observations;
  };

  std::vector<Prior> priors;
  double lambda_new{0.95};
  double lambda_stable{0.75};
  double hour_tolerance{1.5};
  int max_contexts_per_type{3};
  int max_observations_per_context{10};

  static MealType classify_meal_type(double hour) {
    if (hour >= 5 && hour < 10) return MealType::BREAKFAST;
    if (hour >= 11 && hour < 14) return MealType::LUNCH;
    if (hour >= 17 && hour < 21) return MealType::DINNER;
    return MealType::SNACK;
  }

  double hour_of_day(double t_min) const {
    return std::fmod(t_min, 24.0 * 60.0) / 60.0;
  }

  static double circular_hour_distance(double a, double b) {
    double diff = std::abs(a - b);
    return std::min(diff, 24.0 - diff);
  }

  const Prior* best_prior_for(double h, MealType type) const {
    const Prior* best = nullptr;
    double best_dist = std::numeric_limits<double>::infinity();

    for (const auto& p : priors) {
      if (p.type != type || p.n_observations <= 0)
        continue;
      double dist = circular_hour_distance(h, p.hour_of_day);
      if (dist <= hour_tolerance && dist < best_dist) {
        best = &p;
        best_dist = dist;
      }
    }

    if (best != nullptr)
      return best;

    for (const auto& p : priors) {
      if (p.type != type || p.n_observations <= 0)
        continue;
      double dist = circular_hour_distance(h, p.hour_of_day);
      if (dist < best_dist) {
        best = &p;
        best_dist = dist;
      }
    }

    return best;
  }

  void prune_contexts(MealType type, double reference_hour) {
    while (true) {
      int count = 0;
      for (const auto& p : priors)
        if (p.type == type)
          ++count;

      if (count <= max_contexts_per_type)
        return;

      auto worst = priors.end();
      for (auto it = priors.begin(); it != priors.end(); ++it) {
        if (it->type != type)
          continue;
        if (worst == priors.end()) {
          worst = it;
          continue;
        }
        if (it->n_observations < worst->n_observations) {
          worst = it;
          continue;
        }
        if (it->n_observations == worst->n_observations &&
            circular_hour_distance(reference_hour, it->hour_of_day) >
              circular_hour_distance(reference_hour, worst->hour_of_day)) {
          worst = it;
        }
      }

      if (worst == priors.end())
        return;
      priors.erase(worst);
    }
  }

  bool get_prior(double t_min, std::vector<double>& theta_out) const {
    double h = hour_of_day(t_min);
    MealType type = classify_meal_type(h);

    const Prior* best = best_prior_for(h, type);
    if (best != nullptr) {
      theta_out = best->theta_meal;
      return true;
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
      if (it->type == type &&
          circular_hour_distance(h, it->hour_of_day) <= hour_tolerance)
        break;
      ++it;
    }

    if (it == priors.end()) {
      double lambda = lambda_new;
      priors.push_back({type, h, theta_meal, lambda, 1});
      prune_contexts(type, h);
    } else {
      int next_n = std::min(it->n_observations + 1, max_observations_per_context);
      double lambda = (next_n < 3) ? lambda_new :
                      (next_n >= 10) ? lambda_stable :
                      0.85;

      for (size_t i = 0; i < theta_meal.size(); ++i) {
        if (i < it->theta_meal.size())
          it->theta_meal[i] = lambda * it->theta_meal[i] + (1.0 - lambda) * theta_meal[i];
      }
      it->n_observations = next_n;
      it->lambda = lambda;
    }
  }

  int size() const { return static_cast<int>(priors.size()); }
};

} // namespace uva_padova
