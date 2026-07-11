#include "uva_padova/integrals.hpp"
#include "uva_padova/constants.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>

namespace uva_padova {

// ===========================================================================
//  Apply a discrete meal impulse.
// ===========================================================================
void RK4Integrator::apply_meal_event(State& state, double cho_mg) {
  if (cho_mg > 0.0) {
    state.Qsto1 += cho_mg;
    state.mealD = cho_mg;
  }
}

// ===========================================================================
//  Algebraic steady-state solve.
//
//  Given u_basal [pmol/kg/min], compute full SS state by solving the
//  stationary equations algebraically, WITHOUT RK4 integration.
//
//  Returns G_plasma [mg/dL].
// ===========================================================================
static double steady_state_solve(State& state_out, const Params& p,
                                  double u_basal) {
  // --- SC insulin compartments ---
  double Isc1 = u_basal / (p.kd + p.ka1);
  double Isc2 = p.kd * Isc1 / p.ka2;

  // --- Insulin kinetics: solve 2×2 system for Ip, Ii ---
  // 0 = -(m2+m4)*Ip + m1*Ii + ka1*Isc1 + ka2*Isc2
  // 0 = -(m1+m4)*Ii + m2*Ip
  double denom = (p.m2 + p.m4) * (p.m1 + p.m4) - p.m1 * p.m2;
  if (std::abs(denom) < 1e-20) denom = 1e-20;
  double rhs_I = p.ka1 * Isc1 + p.ka2 * Isc2;
  double Ip = rhs_I * (p.m1 + p.m4) / denom;
  double Ii = p.m2 * Ip / (p.m1 + p.m4);

  // --- EGP delays (SS: I_prime = Xl = Ip/VI) ---
  double I_conc = Ip / p.VI;
  double I_prime = I_conc;
  double Xl = I_prime;

  // --- Utilization delay (SS: X = I_conc - Ib) ---
  double X = I_conc - p.Ib;

  // --- Risk function at target glucose ---
  double G_conc = p.Gb;
  double risk = 0.0;
  if (p.r1 > 0.0 && G_conc < p.Gb) {
    using std::log; using std::pow;
    double r2 = 2.0;
    double f_val = pow(log(G_conc), r2) - pow(log(p.Gb), r2);
    if (G_conc < p.Gth) {
      f_val = pow(log(p.Gth), r2) - pow(log(p.Gb), r2);
    }
    risk = 10.0 * f_val * f_val;
  }
  double Vm = p.Vm0 + p.Vmx * X * (1.0 + p.r1 * risk);
  if (Vm < 0.0) Vm = 0.0;

  // --- Glucose subsystem: 2×2 Newton on (Gp, Gt) ---
  double Gp = p.Gb * p.VG;
  double Gt = Gp;

  // Precompute constants
  double EGP_headroom = p.kp1 - p.kp3 * Xl;  // Gp-independent part of EGP

  for (int iter = 0; iter < 15; ++iter) {
    // EGP
    double EGP_val = EGP_headroom - p.kp2 * Gp;
    if (EGP_val < 0.0) EGP_val = 0.0;

    // Renal excretion
    double E_val = (Gp > p.ke2) ? p.ke1 * (Gp - p.ke2) : 0.0;

    // Insulin-dependent utilization
    double Uid = Vm * Gt / (p.Km0 + Gt);

    // Residuals
    double r0 = EGP_val - p.Fcns - E_val - p.k1 * Gp + p.k2 * Gt;
    double r1 = -Uid + p.k1 * Gp - p.k2 * Gt;

    // Jacobian
    double dEGP_dGp = (EGP_val > 0.0) ? -p.kp2 : 0.0;
    double dE_dGp  = (Gp > p.ke2) ? p.ke1 : 0.0;
    double J00 = dEGP_dGp - dE_dGp - p.k1;
    double J01 = p.k2;
    double J10 = p.k1;
    double dUid_dGt = Vm * p.Km0 / ((p.Km0 + Gt) * (p.Km0 + Gt));
    double J11 = -dUid_dGt - p.k2;

    // Solve 2×2
    double det = J00 * J11 - J01 * J10;
    if (std::abs(det) < 1e-30) det = (det >= 0.0) ? 1e-30 : -1e-30;
    double dGp = (-J11 * r0 + J01 * r1) / det;
    double dGt = (J10 * r0 - J00 * r1) / det;

    Gp += dGp;
    Gt += dGt;

    // Safety clamps
    if (Gp < 1.0) Gp = 1.0;
    if (Gt < 1.0) Gt = 1.0;

    if (std::abs(dGp) < 1e-10 && std::abs(dGt) < 1e-10) break;
  }

  // --- Fill output state ---
  state_out = State{};
  state_out.Gp     = Gp;
  state_out.Gt     = Gt;
  state_out.Ip     = Ip;
  state_out.Ii     = Ii;
  state_out.Isc1   = Isc1;
  state_out.Isc2   = Isc2;
  state_out.I_prime = I_prime;
  state_out.Xl     = Xl;
  state_out.X      = X;
  state_out.Gsc    = Gp / p.VG;
  state_out.Qsto1  = 0.0;
  state_out.Qsto2  = 0.0;
  state_out.Qgut   = 0.0;
  state_out.mealD  = 0.0;

  return Gp / p.VG;
}

// ===========================================================================
//  Steady-state solver — bisection on basal insulin to hit Gb.
//  Uses algebraic SS solve (no RK4 warmup).
//  Returns the insulin infusion rate [pmol/kg/min] that achieves Gb.
// ===========================================================================
double SteadyStateSolver::find(State& state_out, const Params& p,
                                double target_Gb) {
  if (target_Gb < 0.0) target_Gb = p.Gb;

  // --- bracket ---
  double insulin_lo = 0.0;
  State lo_state;
  double G_lo = steady_state_solve(lo_state, p, insulin_lo);

  double insulin_hi = 10.0;
  State hi_state;
  double G_hi = steady_state_solve(hi_state, p, insulin_hi);

  if (G_lo <= target_Gb) { state_out = lo_state; return insulin_lo; }
  if (G_hi >= target_Gb) { state_out = hi_state; return insulin_hi; }

  // --- bisection ---
  constexpr int kMaxIter = 50;
  for (int iter = 0; iter < kMaxIter; ++iter) {
    double insulin_mid = 0.5 * (insulin_lo + insulin_hi);
    State mid_state;
    double G_mid = steady_state_solve(mid_state, p, insulin_mid);

    if (std::abs(G_mid - target_Gb) < 1e-9) {
      state_out = mid_state;
      return insulin_mid;
    }
    if (G_mid > target_Gb) {
      insulin_lo = insulin_mid;
      lo_state = mid_state;
    } else {
      insulin_hi = insulin_mid;
      hi_state = mid_state;
    }
  }

  double insulin_final = 0.5 * (insulin_lo + insulin_hi);
  steady_state_solve(state_out, p, insulin_final);
  return insulin_final;
}

// ===========================================================================
//  Forward warmup — kept for backward compatibility (used in drift test etc.)
// ===========================================================================
double SteadyStateSolver::forward_warmup(State& state, const Params& p,
                                          double insulin_rate,
                                          int duration_min) {
  Inputs u;
  u.insulin_rate = insulin_rate;

  for (int t = 0; t < duration_min; ++t) {
    RK4Integrator::step(UvaPadovaODESystem{}, state, u, p, 1.0);
  }

  return UvaPadovaODESystem::plasma_glucose(state, p);
}

} // namespace uva_padova
