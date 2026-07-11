#pragma once

namespace uva_padova {

// ---------------------------------------------------------------------------
// Uniform delivery utility: rate(t) = D/T  for t in [t_start, t_start+T)
//
//   T > 0 :  uniform delivery at rate D/T over the window
//   T <= 0 :  instantaneous (delta) — returns D at t == t_start, else 0
//
// Usage:
//   // 50g CHO spread over 15 minutes:
//   state.Qsto1 += UniformDelivery::rate_at(t, meal_time, 50000.0, 15.0);
//
//   // 6U bolus (480 pmol/kg/min for 1 min — instantaneous):
//   u.insulin_rate += UniformDelivery::rate_at(t, bolus_time, 480.0, 1.0);
//
//   // 6U bolus (same dose spread over 30 min):
//   u.insulin_rate += UniformDelivery::rate_at(t, bolus_time, 480.0, 30.0);
// ---------------------------------------------------------------------------
struct UniformDelivery {
  static double rate_at(double t, double t_start, double D, double T) {
    if (T <= 0.0) {
      // Instantaneous (delta impulse)
      return (t == t_start) ? D : 0.0;
    }
    // Uniform over [t_start, t_start+T)
    if (t < t_start || t >= t_start + T) return 0.0;
    return D / T;
  }
};

} // namespace uva_padova
