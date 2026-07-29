#pragma once

#include <cstdint>
#include <cmath>
#include <string>
#include <type_traits>
#include <stdexcept>

namespace uva_padova {

// ---------------------------------------------------------------------------
// Parameter ID enum — single source of truth for param name ↔ field mapping
// ---------------------------------------------------------------------------
enum class ParamId : int {
  VG, k1, k2, VI, m1, m2, m4, kmax, kmin, kabs, kgri, f, b, d,
  kp1, kp2, kp3, ki, Fcns, Vm0, Vmx, Km0, p2U, r1, Gth,
  ke1, ke2, ka1, ka2, kd, Ts, BW, Gb, Ib,
  HE_b, r2,
  COUNT
};

inline const std::string& param_name(ParamId id) {
  static const std::string names[] = {
    "VG", "k1", "k2", "VI", "m1", "m2", "m4", "kmax", "kmin",
    "kabs", "kgri", "f", "b", "d",
    "kp1", "kp2", "kp3", "ki", "Fcns", "Vm0", "Vmx", "Km0",
    "p2U", "r1", "Gth",
    "ke1", "ke2", "ka1", "ka2", "kd",
    "Ts", "BW", "Gb", "Ib",
    "HE_b", "r2"
  };
  return names[static_cast<int>(id)];
}

inline ParamId param_id_from_string(const std::string& name) {
  static const char* names[] = {
    "VG", "k1", "k2", "VI", "m1", "m2", "m4", "kmax", "kmin",
    "kabs", "kgri", "f", "b", "d",
    "kp1", "kp2", "kp3", "ki", "Fcns", "Vm0", "Vmx", "Km0",
    "p2U", "r1", "Gth",
    "ke1", "ke2", "ka1", "ka2", "kd",
    "Ts", "BW", "Gb", "Ib",
    "HE_b", "r2"
  };
  for (int i = 0; i < static_cast<int>(ParamId::COUNT); ++i) {
    if (name == names[i]) return static_cast<ParamId>(i);
  }
  throw std::out_of_range("unknown parameter name: " + name);
}

// ---------------------------------------------------------------------------
// Primary state vector for the reduced T1DM single-hormone model.
// Templated on scalar type T to support autodiff (ceres::Jet).
// ---------------------------------------------------------------------------
template<typename T>
struct BasicState {
  static constexpr int kGpIndex = 0;
  static constexpr int kGtIndex = 1;
  static constexpr int kQsto1Index = 2;
  static constexpr int kQsto2Index = 3;
  static constexpr int kQgutIndex = 4;
  static constexpr int kIpIndex = 5;
  static constexpr int kIiIndex = 6;
  static constexpr int kIsc1Index = 7;
  static constexpr int kIsc2Index = 8;
  static constexpr int kIPrimeIndex = 9;
  static constexpr int kXlIndex = 10;
  static constexpr int kXIndex = 11;
  static constexpr int kGscIndex = 12;
  static constexpr int kMealDIndex = 13;
  static constexpr int kSize = 14;

  T Gp{0};      // plasma glucose mass          [mg/kg]
  T Gt{0};      // tissue glucose mass          [mg/kg]
  T Qsto1{0};   // stomach solid phase          [mg]
  T Qsto2{0};   // stomach liquid phase         [mg]
  T Qgut{0};    // gut (intestine)              [mg]
  T Ip{0};      // plasma insulin               [pmol/kg]
  T Ii{0};      // hepatic insulin              [pmol/kg]
  T Isc1{0};    // SC insulin fast              [pmol/kg]
  T Isc2{0};    // SC insulin slow              [pmol/kg]
  T I_prime{0}; // intermediate insulin signal  [pmol/L]
  T Xl{0};      // delayed EGP insulin signal   [dimensionless]
  T X{0};       // insulin action on utilization [dimensionless]
  T Gsc{0};     // subcutaneous glucose (CGM)    [mg/dl]
  T mealD{0};   // fixed CHO mass of active meal [mg]

  T& at(int i);
  const T& at(int i) const;

  // Runtime checks — only meaningful for double
  bool all_finite() const;
  void clamp_physiological();
};

// ---------------------------------------------------------------------------
// External inputs to the model at a given time step.
// ---------------------------------------------------------------------------
template<typename T>
struct BasicInputs {
  T insulin_rate{0}; // insulin infusion          [pmol/kg/min]
};

// ---------------------------------------------------------------------------
// Parameters of the UVA-Padova model (reduced T1DM set).
// ---------------------------------------------------------------------------
template<typename T>
struct BasicParams {
  // --- Glucose kinetics [A1] ---
  T VG{1.88};    // glucose distribution volume      [dl/kg]
  T k1{0.065};   // transfer Gp -> Gt                [1/min]
  T k2{0.079};   // transfer Gt -> Gp                [1/min]

  // --- Insulin kinetics [A2] ---
  T VI{0.05};    // insulin distribution volume      [L/kg]
  T m1{0.190};   // hepatic insulin transfer         [1/min]
  T m2{0.484};   // plasma insulin clearance         [1/min]
  T m4{0.194};   // plasma insulin degradation       [1/min]
  T HE_b{0.6};   // hepatic extraction, constant T1DM [dimensionless]

  // --- GI absorption [A3–A4] ---
  T kmax{0.0558}; // max gastric emptying rate       [1/min]
  T kmin{0.0080}; // min gastric emptying rate       [1/min]
  T kabs{0.057};  // intestinal absorption rate      [1/min]
  T kgri{0.0558}; // solid/liquid grinding rate      [1/min]
  T f{0.90};      // fraction of CHO absorbed        [dimensionless]
  T b{0.82};      // gastric shape param             [dimensionless]
  T d{0.010};     // gastric shape param             [dimensionless]

  // --- Endogenous glucose production [A5–A8] ---
  T kp1{2.70};    // basal EGP                       [mg/kg/min]
  T kp2{0.0021};  // glucose suppression of EGP      [1/min]
  T kp3{0.009};   // insulin suppression of EGP      [mg/kg/min per pmol/L]
  T ki{0.0079};   // EGP delay time constant         [1/min]

  // --- Utilization [A9–A13] ---
  T Fcns{1.0};    // CNS glucose uptake              [mg/kg/min]
  T Vm0{2.50};    // basal utilization rate          [mg/kg/min]
  T Vmx{0.047};   // insulin-sensitive util. rate    [mg/kg/min per pmol/L]
  T Km0{225.59};  // Michaelis constant               [mg/kg]
  T p2U{0.0331};  // utilization delay               [1/min]
  T r1{0.0};      // hypo risk multiplier            [dimensionless]
  T r2{2.0};      // risk function shape exponent    [dimensionless]
  T Gth{60.0};    // hypo threshold                  [mg/dl]

  // --- Renal excretion [A14] ---
  T ke1{0.0005};  // excretion rate                  [1/min]
  T ke2{339.0};   // renal threshold                 [mg/kg]

  // --- SC insulin absorption [A15–A17] ---
  T ka1{0.0057};  // SC fast absorption              [1/min]
  T ka2{0.0057};  // SC slow absorption              [1/min]
  T kd{0.014};    // SC redistribution               [1/min]

  // --- CGM sensor [A24] ---
  T Ts{15.0};     // sensor time constant            [min]

  // --- Subject ---
  T BW{75.0};     // body weight                     [kg]
  T Gb{120.0};    // basal glucose (fasting)         [mg/dl]
  T Ib{70.0};     // basal plasma insulin            [pmol/L]

  static BasicParams defaults_t1d() { return {}; }
};

// --- Convenience aliases (double precision) ---
using State   = BasicState<double>;
using Inputs  = BasicInputs<double>;
using Params  = BasicParams<double>;

// ---------------------------------------------------------------------------
// ParamId → BasicParams<T> field access — single source of truth
// ---------------------------------------------------------------------------
template<typename T>
inline T& param_by_id(BasicParams<T>& p, ParamId id) {
  switch (id) {
    case ParamId::VG:   return p.VG;   case ParamId::k1:   return p.k1;   case ParamId::k2:   return p.k2;
    case ParamId::VI:   return p.VI;   case ParamId::m1:   return p.m1;   case ParamId::m2:   return p.m2;
    case ParamId::m4:   return p.m4;   case ParamId::HE_b: return p.HE_b;
    case ParamId::kmax: return p.kmax; case ParamId::kmin: return p.kmin;
    case ParamId::kabs: return p.kabs; case ParamId::kgri: return p.kgri;
    case ParamId::f:    return p.f;    case ParamId::b:    return p.b;    case ParamId::d:    return p.d;
    case ParamId::kp1:  return p.kp1;  case ParamId::kp2:  return p.kp2;  case ParamId::kp3:  return p.kp3;
    case ParamId::ki:   return p.ki;
    case ParamId::Fcns: return p.Fcns; case ParamId::Vm0:  return p.Vm0;  case ParamId::Vmx:  return p.Vmx;
    case ParamId::Km0:  return p.Km0;  case ParamId::p2U:  return p.p2U;
    case ParamId::r1:   return p.r1;   case ParamId::r2:   return p.r2;   case ParamId::Gth:  return p.Gth;
    case ParamId::ke1:  return p.ke1;  case ParamId::ke2:  return p.ke2;
    case ParamId::ka1:  return p.ka1;  case ParamId::ka2:  return p.ka2;  case ParamId::kd:   return p.kd;
    case ParamId::Ts:   return p.Ts;
    case ParamId::BW:   return p.BW;   case ParamId::Gb:   return p.Gb;   case ParamId::Ib:   return p.Ib;
    default: __builtin_trap();
  }
}

template<typename T>
inline const T& param_by_id(const BasicParams<T>& p, ParamId id) {
  return param_by_id(const_cast<BasicParams<T>&>(p), id);
}

// Convenience: double-only extractor
inline double param_value(const Params& p, ParamId id) {
  return param_by_id(p, id);
}
inline double param_value(const Params& p, const std::string& name) {
  return param_value(p, param_id_from_string(name));
}

// ---------------------------------------------------------------------------
// Observation: CGM reading (subcutaneous glucose)
// ---------------------------------------------------------------------------
struct Observation {
  double Gsc{0.0};
  double time_min{0.0};
};

// ---------------------------------------------------------------------------
// Data records for data ingestion
// ---------------------------------------------------------------------------
struct MealRecord {
  double time_min{0.0};
  double cho_mg{0.0};
};

struct InsulinRecord {
  double time_min{0.0};
  double rate_pmol_per_kg_per_min{0.0};
};

struct CGMRecord {
  double time_min{0.0};
  double mg_dl{0.0};
};

// ===========================================================================
//  Template method implementations
// ===========================================================================

template<typename T>
T& BasicState<T>::at(int i) {
  switch (i) {
    case 0:  return Gp;
    case 1:  return Gt;
    case 2:  return Qsto1;
    case 3:  return Qsto2;
    case 4:  return Qgut;
    case 5:  return Ip;
    case 6:  return Ii;
    case 7:  return Isc1;
    case 8:  return Isc2;
    case 9:  return I_prime;
    case 10: return Xl;
    case 11: return X;
    case 12: return Gsc;
    case 13: return mealD;
    default: __builtin_trap();  // out of bounds
  }
}

template<typename T>
const T& BasicState<T>::at(int i) const {
  return const_cast<BasicState*>(this)->at(i);
}

template<typename T>
bool BasicState<T>::all_finite() const {
  if constexpr (std::is_same_v<T, double>) {
    return std::isfinite(Gp) && std::isfinite(Gt) &&
           std::isfinite(Qsto1) && std::isfinite(Qsto2) && std::isfinite(Qgut) &&
           std::isfinite(Ip) && std::isfinite(Ii) &&
           std::isfinite(Isc1) && std::isfinite(Isc2) &&
           std::isfinite(I_prime) && std::isfinite(Xl) && std::isfinite(X) &&
           std::isfinite(Gsc) && std::isfinite(mealD);
  }
  return true; // Jet types are always "finite"
}

template<typename T>
void BasicState<T>::clamp_physiological() {
  // Smooth clamp for all types: differentiable softplus-based saturation.
  //   smooth_clamp(v, lo, hi) = v - softplus(v - hi) + softplus(lo - v)
  // Keeps double and Jet paths identical — consistent cost, clean gradients.
  using std::log; using std::exp;
  auto softplus = [](const T& x) -> T {
    if (x > T(30)) return x;
    if (x < T(-30)) return T(0);
    return log(T(1) + exp(x));
  };
  auto clamp_val = [&](T& v, T lo, T hi) {
    v = v - softplus(v - hi) + softplus(lo - v);
  };
  clamp_val(Gp,    T(0),    T(1000));
  clamp_val(Gt,    T(0),    T(1000));
  clamp_val(Qsto1, T(0),    T(300000));
  clamp_val(Qsto2, T(0),    T(300000));
  clamp_val(Qgut,  T(0),    T(300000));
  clamp_val(Ip,    T(0),    T(5000));
  clamp_val(Ii,    T(0),    T(5000));
  clamp_val(Isc1,  T(0),    T(5000));
  clamp_val(Isc2,  T(0),    T(5000));
  clamp_val(I_prime,T(0),   T(5000));
  clamp_val(Xl,    T(0),    T(100));
  clamp_val(X,     T(-100), T(100));
  clamp_val(Gsc,   T(0),    T(1000));
  clamp_val(mealD, T(0),    T(300000));
}

} // namespace uva_padova
