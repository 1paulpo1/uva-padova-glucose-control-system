#pragma once

#include "uva_padova/cost_function.hpp"
#include <vector>

namespace uva_padova {

// Build an IdentificationData for testing.
// observation_times: CGM observation times [min]
// meal_times/meal_chos: paired meal events
// bolus_times/bolus_doses: paired bolus events
// cgm_value_fill: constant CGM value to fill all observations
inline IdentificationData make_id_data(
    int total_minutes,
    const std::vector<double>& observation_times,
    const std::vector<double>& meal_times = {},
    const std::vector<double>& meal_chos = {},
    const std::vector<double>& bolus_times = {},
    const std::vector<double>& bolus_doses = {},
    double cgm_value_fill = 120.0) {
  IdentificationData d;
  d.cgm_time = observation_times;
  d.cgm_value.assign(observation_times.size(), cgm_value_fill);
  d.meal_time = meal_times;
  d.meal_cho = meal_chos;
  d.bolus_time = bolus_times;
  d.bolus_dose = bolus_doses;
  d.finalize();
  return d;
}

} // namespace uva_padova
