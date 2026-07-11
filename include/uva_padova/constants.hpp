#pragma once

#include <type_traits>

namespace uva_padova {
namespace constants {

// Molar mass of glucose
inline constexpr double kGlucoseMolarMass = 180.156; // g/mol

// Conversion: 1 mmol/L = 18.0156 mg/dL
inline constexpr double kMmolPerL_To_MgPerDl = 18.0156;

// Conversion: mg/dL -> mmol/L
inline constexpr double kMgPerDl_To_MmolPerL = 1.0 / 18.0156;

// Insulin: 1 U (unit) = 6000 pmol (standard conversion)
inline constexpr double kInsulinUnits_To_Pmol = 6000.0;

// 1 U/hour -> pmol/kg/min (needs body weight)
inline double u_per_hr_to_pmol_per_kg_per_min(double u_per_hr, double BW_kg) {
  return (u_per_hr * kInsulinUnits_To_Pmol) / (60.0 * BW_kg);
}

// Safe division for any scalar type T
template<typename T>
inline T safe_div(const T& num, const T& den, const T& fallback = T(0)) {
  return (den > T(1e-12)) ? num / den : fallback;
}

// Clamp for any scalar type T
template<typename T>
inline T clamp(const T& val, const T& lo, const T& hi) {
  if (val < lo) return lo;
  if (val > hi) return hi;
  return val;
}

} // namespace constants
} // namespace uva_padova