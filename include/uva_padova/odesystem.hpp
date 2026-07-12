#pragma once

#include "types.hpp"
#include "constants.hpp"
#include <cmath>

namespace uva_padova {

// ---------------------------------------------------------------------------
// UVA-Padova ODE right-hand side (reduced T1DM single-hormone model).
// Templated on scalar type T to support autodiff (ceres::Jet via ADL).
//
// Math function calls use the "using std::...; func(x)" pattern so that
// ADL finds ceres overloads for Jet types.
// ---------------------------------------------------------------------------
struct UvaPadovaODESystem {

  // Evaluate the RHS at a given state.
  template<typename T>
  static void rhs(const BasicState<T>& x, BasicState<T>& dxdt,
                  const BasicInputs<T>& u, const BasicParams<T>& p) {
    using std::tanh; using std::exp; using std::log; using std::pow;
    using std::abs;

    dxdt = {};

    // ---- Glucose kinetics [A1] ----
    {
      T G = x.Gp / p.VG;
      T risk = risk_factor(G, p.Gb, p.Gth, p.r1);
      T Vm  = p.Vm0 + p.Vmx * x.X * (T(1) + p.r1 * risk);
      if (Vm < T(0)) Vm = T(0);
      T Uid = Vm * x.Gt / (p.Km0 + x.Gt);
      T Ra_meal = p.f * p.kabs * x.Qgut / p.BW;
      T EGP_val = p.kp1 - p.kp2 * x.Gp - p.kp3 * x.Xl;
      if (EGP_val < T(0)) EGP_val = T(0);
      T E_val = (x.Gp > p.ke2) ? p.ke1 * (x.Gp - p.ke2) : T(0);

      dxdt.Gp = EGP_val + Ra_meal - p.Fcns - E_val - p.k1 * x.Gp + p.k2 * x.Gt;
      dxdt.Gt = -Uid + p.k1 * x.Gp - p.k2 * x.Gt;
    }

    // ---- Insulin kinetics [A2] ----
    {
      T Ra_I = p.ka1 * x.Isc1 + p.ka2 * x.Isc2;
      dxdt.Ip = -(p.m2 + p.m4) * x.Ip + p.m1 * x.Ii + Ra_I;
      dxdt.Ii = -(p.m1 + p.m4) * x.Ii + p.m2 * x.Ip;
    }

    // ---- GI absorption [A3–A4] ----
    {
      T total_Qsto = x.Qsto1 + x.Qsto2;
      T k_emp = gastric_emptying_rate(total_Qsto, x.mealD, p);
      dxdt.Qsto1 = -p.kgri * x.Qsto1;
      dxdt.Qsto2 = -k_emp * x.Qsto2 + p.kgri * x.Qsto1;
      dxdt.Qgut  = -p.kabs * x.Qgut + k_emp * x.Qsto2;
    }

    // ---- EGP delays [A6–A7] ----
    {
      T I_conc = constants::safe_div(x.Ip, p.VI);
      dxdt.I_prime = -p.ki * (x.I_prime - I_conc);
      dxdt.Xl = -p.ki * (x.Xl - x.I_prime);
    }

    // ---- Utilization delay [A11] ----
    {
      T I_conc = constants::safe_div(x.Ip, p.VI);
      dxdt.X = -p.p2U * x.X + p.p2U * (I_conc - p.Ib);
    }

    // ---- SC insulin absorption [A15–A17] ----
    {
      dxdt.Isc1 = -(p.kd + p.ka1) * x.Isc1 + u.insulin_rate;
      dxdt.Isc2 = p.kd * x.Isc1 - p.ka2 * x.Isc2;
    }

    // ---- CGM sensor [A24] ----
    {
      T G_conc = x.Gp / p.VG;
      dxdt.Gsc = -(T(1) / p.Ts) * x.Gsc + (T(1) / p.Ts) * G_conc;
    }

    // mealD derivative is 0 (fixed at meal start)
  }

  // Convenience: return dxdt as a State object
  template<typename T>
  static BasicState<T> rhs(const BasicState<T>& x,
                           const BasicInputs<T>& u,
                           const BasicParams<T>& p) {
    BasicState<T> dxdt;
    rhs(x, dxdt, u, p);
    return dxdt;
  }

  // ---- Individual equation blocks (exposed for testing) ----

  template<typename T>
  static void glucose_kinetics(const BasicState<T>& x, BasicState<T>& dxdt,
                                const BasicParams<T>& p) {
    BasicInputs<T> dummy; BasicState<T> tmp;
    rhs(x, tmp, dummy, p);
    dxdt.Gp = tmp.Gp; dxdt.Gt = tmp.Gt;
  }

  template<typename T>
  static void insulin_kinetics(const BasicState<T>& x, BasicState<T>& dxdt,
                                const BasicParams<T>& p) {
    BasicInputs<T> dummy; BasicState<T> tmp;
    rhs(x, tmp, dummy, p);
    dxdt.Ip = tmp.Ip; dxdt.Ii = tmp.Ii;
  }

  template<typename T>
  static void gi_absorption(const BasicState<T>& x, BasicState<T>& dxdt,
                             const BasicParams<T>& p) {
    BasicInputs<T> dummy; BasicState<T> tmp;
    rhs(x, tmp, dummy, p);
    dxdt.Qsto1 = tmp.Qsto1; dxdt.Qsto2 = tmp.Qsto2; dxdt.Qgut = tmp.Qgut;
  }

  template<typename T>
  static void egp_delays(const BasicState<T>& x, BasicState<T>& dxdt,
                          const BasicParams<T>& p) {
    BasicInputs<T> dummy; BasicState<T> tmp;
    rhs(x, tmp, dummy, p);
    dxdt.I_prime = tmp.I_prime; dxdt.Xl = tmp.Xl;
  }

  template<typename T>
  static void utilization_delay(const BasicState<T>& x, BasicState<T>& dxdt,
                                 const BasicParams<T>& p) {
    BasicInputs<T> dummy; BasicState<T> tmp;
    rhs(x, tmp, dummy, p);
    dxdt.X = tmp.X;
  }

  template<typename T>
  static void sc_insulin(const BasicState<T>& x, BasicState<T>& dxdt,
                          const BasicInputs<T>& u, const BasicParams<T>& p) {
    BasicState<T> tmp;
    rhs(x, tmp, u, p);
    dxdt.Isc1 = tmp.Isc1; dxdt.Isc2 = tmp.Isc2;
  }

  template<typename T>
  static void cgm_sensor(const BasicState<T>& x, BasicState<T>& dxdt,
                          const BasicParams<T>& p) {
    BasicInputs<T> dummy; BasicState<T> tmp;
    rhs(x, tmp, dummy, p);
    dxdt.Gsc = tmp.Gsc;
  }

  // ---- Helpers ----

  template<typename T>
  static T plasma_glucose(const BasicState<T>& x, const BasicParams<T>& p) {
    return x.Gp / p.VG;
  }

  template<typename T>
  static T egp(const BasicState<T>& x, const BasicParams<T>& p) {
    using std::abs;
    T val = p.kp1 - p.kp2 * x.Gp - p.kp3 * x.Xl;
    return (val < T(0)) ? T(0) : val;
  }

  template<typename T>
  static T renal_excretion(const T& Gp, const BasicParams<T>& p) {
    return (Gp > p.ke2) ? p.ke1 * (Gp - p.ke2) : T(0);
  }

  template<typename T>
  static T utilization_insulin_dependent(const BasicState<T>& x,
                                          const BasicParams<T>& p) {
    T G = plasma_glucose(x, p);
    T risk = risk_factor(G, p.Gb, p.Gth, p.r1);
    T Vm = p.Vm0 + p.Vmx * x.X * (T(1) + p.r1 * risk);
    if (Vm < T(0)) Vm = T(0);
    return Vm * x.Gt / (p.Km0 + x.Gt);
  }

  // Unified risk computation per validated formula (A13):
  //   risk = 10 · [log(G/Gb)]^4  for G < Gb
  //   risk = 10 · [log(Gth/Gb)]^4 for G < Gth
  // G is clamped to ≥1e-3 mg/dL to protect log().
  template<typename T>
  static T risk_factor(const T& G_mgdl, const T& Gb,
                        const T& Gth, const T& r1) {
    using std::log;
    if (r1 == T(0)) return T(0);
    if (G_mgdl >= Gb) return T(0);
    T G_safe = (G_mgdl > T(1e-3)) ? G_mgdl : T(1e-3);
    T ratio = log(G_safe / Gb);
    T f_val = ratio * ratio;
    if (G_mgdl < Gth) {
      T ratio_th = log(Gth / Gb);
      f_val = ratio_th * ratio_th;
    }
    return T(10) * f_val * f_val;
  }

  template<typename T>
  static T risk_function(const T& G_mgdl, const T& Gb,
                          const T& Gth, const T& r1) {
    return risk_factor(G_mgdl, Gb, Gth, r1);
  }

  template<typename T>
  static T f_risk(const T& G, const T& Gb, const T& r2) {
    using std::log; using std::pow;
    if (G <= T(0) || Gb <= T(0)) return T(0);
    T G_safe = (G > T(1e-3)) ? G : T(1e-3);
    T ratio = log(G_safe / Gb);
    return pow(ratio, r2);
  }

  template<typename T>
  static T gastric_emptying_rate(const T& Qsto, const T& D,
                                  const BasicParams<T>& p) {
    using std::tanh;
    if (D <= T(0)) return p.kmax;
    T alpha = T(5) / (T(2) * D * (T(1) - p.b));
    T beta  = T(5) / (T(2) * D * p.d);
    T arg1 = alpha * (Qsto - p.b * D);
    T arg2 = beta  * (Qsto - p.d * D);
    return p.kmin + (p.kmax - p.kmin) * T(0.5) *
           (tanh(arg1) - tanh(arg2) + T(2));
  }
};

} // namespace uva_padova
