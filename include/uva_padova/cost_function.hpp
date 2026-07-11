#pragma once

#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <numeric>

#include <Eigen/Core>
#include <ceres/jet.h>

#include <ceres/cost_function.h>

#include "uva_padova/types.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/integrals.hpp"
#include "uva_padova/delivery.hpp"
#include "uva_padova/constants.hpp"

namespace uva_padova {

// ---------------------------------------------------------------------------
// Data bundle for offline identification
// ---------------------------------------------------------------------------
struct IdentificationData {
  std::vector<double> cgm_time;     // [min]
  std::vector<double> cgm_value;    // [mg/dL]
  std::vector<double> meal_time;    // [min]
  std::vector<double> meal_cho;     // [mg]
  std::vector<double> insulin_time; // [min]
  std::vector<double> insulin_rate; // [pmol/kg/min]
  std::vector<double> bolus_time;   // [min] — subcutaneous bolus injections
  std::vector<double> bolus_dose;   // [pmol/kg] — injected into Isc1

  // Derived
  int total_min_{0};

  void finalize() {
    total_min_ = static_cast<int>(cgm_time.empty() ? 0 : cgm_time.back());
    if (!cgm_time.empty())
      total_min_ = std::max(total_min_, static_cast<int>(std::ceil(cgm_time.back())));
  }

  int total_min() const { return total_min_; }
};

// ---------------------------------------------------------------------------
// Parameter mapping descriptor
// ---------------------------------------------------------------------------
struct ParamDescriptor {
  std::string name;
  double prior_mean{1.0};
  double prior_std{2.0};
  double initial_val{1.0};
};

// ---------------------------------------------------------------------------
// Observation cost — returns sqrt(2·log(cosh(r))) · sign(r) residuals for LM.
//
//   r = (G_sim − G_obs) / σ
//   residual = sign(r) · √(2·log(cosh(r)))
//
// Then ½·Σ residual² = Σ log(cosh(r))  — exact log-cosh robust cost.
//
// Parameter vector θ lives in LOG-space:
//   true_value = base_value · exp(θᵢ)
// ---------------------------------------------------------------------------
struct ObservationCost : ceres::CostFunction {
  static constexpr int kMaxJetDim = 20;
  using JetT = ceres::Jet<double, kMaxJetDim>;

  const IdentificationData& data_;
  const Params base_params_;
  const std::vector<ParamDescriptor>& param_desc_;
  const int n_params_;
  const int n_obs_;
  mutable int eval_count_{0};
  const double sigma_{2.9};

  ObservationCost(const IdentificationData& data,
                  const Params& base_params,
                  const std::vector<ParamDescriptor>& param_desc)
    : data_(data), base_params_(base_params),
      param_desc_(param_desc), n_params_(static_cast<int>(param_desc.size())),
      n_obs_(static_cast<int>(data.cgm_time.size())) {
    if (n_params_ > kMaxJetDim) {
      std::cerr << "ObservationCost: n_params=" << n_params_
                << " exceeds kMaxJetDim=" << kMaxJetDim << "\n";
      std::abort();
    }
    set_num_residuals(n_obs_);
    mutable_parameter_block_sizes()->push_back(n_params_);
  }

  // ---- sqrt-trick: sign(r) · √(2·log(cosh(r))) ----
  // Numerically: log(cosh(r)) = |r| + log1p(exp(-2|r|)) - ln(2)
  template<typename T>
  static T robust_logcosh_sqrt(const T& r) {
    using std::abs; using std::exp; using std::log1p; using std::sqrt;
    double ra = extract_val(r);
    if (std::abs(ra) < 1e-6) {
      return r;  // linear regime: √(2·log(cosh(r))) ↔ r
    }
    T absr = abs(r);
    T logcosh = absr + log1p(exp(T(-2) * absr)) - T(M_LN2);
    return (absr / r) * sqrt(T(2) * logcosh);
  }

  // ---- LM interface (sqrt-trick residuals) ----
  bool Evaluate(const double* const* parameters, double* residuals,
                double** jacobians) const override {
    eval_count_++;
    const double* params = parameters[0];

    if (jacobians != nullptr && jacobians[0] != nullptr) {
      JetT jet_params[kMaxJetDim];
      for (int i = 0; i < n_params_; ++i) {
        jet_params[i].a = params[i];
        jet_params[i].v = Eigen::Matrix<double, kMaxJetDim, 1>::Zero();
        jet_params[i].v[i] = 1.0;
      }
      auto cgm = simulate_observations<JetT>(jet_params);
      for (int i = 0; i < n_obs_; ++i) {
        JetT r = (cgm[i] - JetT(data_.cgm_value[i])) / JetT(sigma_);
        JetT s = robust_logcosh_sqrt(r);
        residuals[i] = extract_val(s);
        for (int j = 0; j < n_params_; ++j)
          jacobians[0][i * n_params_ + j] = extract_deriv(s, j);
      }
    } else {
      auto cgm = simulate_observations<double>(params);
      for (int i = 0; i < n_obs_; ++i) {
        double r = (cgm[i] - data_.cgm_value[i]) / sigma_;
        residuals[i] = robust_logcosh_sqrt(r);
      }
    }
    return true;
  }

  // ---- Raw residuals + Jacobian (no sqrt-trick) for sandwich ----
  // One ODE integration with Jet AD. Returns r_i = (G_sim - G_obs) / σ
  // and J_raw[i,j] = ∂r_i/∂θ_j.
  void compute_raw_gradients(const double* params,
                             std::vector<double>& raw_residuals,
                             Eigen::MatrixXd& J_raw) const {
    eval_count_++;
    JetT jet_params[kMaxJetDim];
    for (int i = 0; i < n_params_; ++i) {
      jet_params[i].a = params[i];
      jet_params[i].v.setZero();
      jet_params[i].v[i] = 1.0;
    }
    auto cgm = simulate_observations<JetT>(jet_params);
    raw_residuals.resize(n_obs_);
    J_raw.resize(n_obs_, n_params_);
    for (int i = 0; i < n_obs_; ++i) {
      JetT r = (cgm[i] - JetT(data_.cgm_value[i])) / JetT(sigma_);
      raw_residuals[i] = extract_val(r);
      for (int j = 0; j < n_params_; ++j)
        J_raw(i, j) = extract_deriv(r, j);
    }
  }

  // ---- Scalar cost with log-cosh (for CMA-ES) ----
  // Returns Σ log(cosh(r_i))  — matches Ceres ½·Σ residual² with sqrt-trick.
  double compute_scalar_cost(const double* params) const {
    using std::log; using std::cosh;
    double total = 0.0;
    auto cgm = simulate_observations<double>(params);
    for (int i = 0; i < n_obs_; ++i) {
      double resid = (cgm[i] - data_.cgm_value[i]) / sigma_;
      total += log(cosh(resid));
    }
    return total;
  }

  // ---- Trajectory simulation ----
  template<typename T>
  std::vector<T> simulate_observations(const T* params) const {
    using std::log; using std::exp; using std::abs;

    BasicParams<T> p;
    copy_params(base_params_, p);
    apply_adjustments(params, p);

    // Steady state
    BasicParams<double> p_double = to_double(p);
    State ss_state;
    double insulin_ss = SteadyStateSolver::find(ss_state, p_double, p_double.Gb);

    // Convert SS to T with sensitivities
    BasicState<T> state;
    T basal_rate(0);

    if constexpr (std::is_same_v<T, double>) {
      for (int i = 0; i < BasicState<T>::kSize; ++i)
        state.at(i) = T(ss_state.at(i));
      basal_rate = T(insulin_ss);
    } else {
      const double eps = 1e-6;

      double theta_double[kMaxJetDim];
      for (int i = 0; i < n_params_; ++i)
        theta_double[i] = extract_val(params[i]);

      Eigen::Matrix<double, kMaxJetDim, 1> di_ss;
      di_ss.setZero();
      std::vector<Eigen::Matrix<double, kMaxJetDim, 1>> dstate(BasicState<T>::kSize);
      for (auto& v : dstate) v.setZero();

      for (int pi = 0; pi < n_params_; ++pi) {
        double saved = theta_double[pi];
        theta_double[pi] += eps;

        BasicParams<double> p_pert;
        copy_params(base_params_, p_pert);
        apply_adjustments(theta_double, p_pert);

        State ss_pert;
        double i_ss_pert = SteadyStateSolver::find(ss_pert, p_pert, p_pert.Gb);

        double inv_eps = 1.0 / eps;
        for (int j = 0; j < BasicState<T>::kSize; ++j)
          dstate[j][pi] = (ss_pert.at(j) - ss_state.at(j)) * inv_eps;
        di_ss[pi] = (i_ss_pert - insulin_ss) * inv_eps;
        theta_double[pi] = saved;
      }

      for (int j = 0; j < BasicState<T>::kSize; ++j)
        state.at(j) = JetT(ss_state.at(j), dstate[j]);
      basal_rate = JetT(insulin_ss, di_ss);
    }

    // Forward simulation
    BasicInputs<T> u;
    u.insulin_rate = basal_rate;

    std::vector<T> cgm_out(n_obs_);
    int meal_idx = 0, ins_idx = 0, obs_idx = 0, bolus_idx = 0;

    for (int t = 0; t <= data_.total_min(); ++t) {
      // Insulin boluses
      while (ins_idx < static_cast<int>(data_.insulin_time.size())) {
        if (static_cast<int>(data_.insulin_time[ins_idx]) == t) {
          u.insulin_rate += T(data_.insulin_rate[ins_idx]);
          ins_idx++;
        } else if (data_.insulin_time[ins_idx] > t) {
          break;
        } else {
          ins_idx++;
        }
      }

      // Bolus injections (subcutaneous, into Isc1)
      while (bolus_idx < static_cast<int>(data_.bolus_time.size())) {
        if (static_cast<int>(data_.bolus_time[bolus_idx]) == t) {
          state.Isc1 += T(data_.bolus_dose[bolus_idx]);
          bolus_idx++;
        } else if (data_.bolus_time[bolus_idx] > t) {
          break;
        } else {
          bolus_idx++;
        }
      }

      // Meal events
      while (meal_idx < static_cast<int>(data_.meal_time.size())) {
        if (static_cast<int>(data_.meal_time[meal_idx]) == t) {
          state.Qsto1 += T(data_.meal_cho[meal_idx]);
          state.mealD = T(data_.meal_cho[meal_idx]);
          meal_idx++;
        } else if (data_.meal_time[meal_idx] > t) {
          break;
        } else {
          meal_idx++;
        }
      }

      // Record CGM at observation times
      while (obs_idx < n_obs_) {
        int obs_t = static_cast<int>(std::round(data_.cgm_time[obs_idx]));
        if (obs_t == t) {
          cgm_out[obs_idx] = state.Gsc;
          obs_idx++;
        } else if (obs_t > t) {
          break;
        } else {
          obs_idx++;
        }
      }

      // RK4 step
      if (t < data_.total_min())
        RK4Integrator::step(UvaPadovaODESystem{}, state, u, p, 1.0);
    }

    return cgm_out;
  }

  int eval_count() const { return eval_count_; }

  // ---- Helpers ----

  static double extract_val(double x) { return x; }
  template<int N>
  static double extract_val(const ceres::Jet<double, N>& x) { return x.a; }

  static double extract_deriv(double /*x*/, int /*idx*/) { return 0.0; }
  template<int N>
  static double extract_deriv(const ceres::Jet<double, N>& x, int idx) { return x.v[idx]; }

  template<typename T>
  static void copy_params(const Params& src, BasicParams<T>& dst) {
    dst.VG   = T(src.VG);   dst.k1  = T(src.k1);   dst.k2  = T(src.k2);
    dst.VI   = T(src.VI);   dst.m1  = T(src.m1);   dst.m2  = T(src.m2);   dst.m4  = T(src.m4);
    dst.kmax = T(src.kmax); dst.kmin = T(src.kmin); dst.kabs = T(src.kabs); dst.kgri = T(src.kgri);
    dst.f    = T(src.f);    dst.b   = T(src.b);     dst.d   = T(src.d);
    dst.kp1  = T(src.kp1);  dst.kp2 = T(src.kp2);   dst.kp3 = T(src.kp3);   dst.ki  = T(src.ki);
    dst.Fcns = T(src.Fcns); dst.Vm0 = T(src.Vm0);   dst.Vmx = T(src.Vmx);
    dst.Km0  = T(src.Km0);  dst.p2U = T(src.p2U);   dst.r1  = T(src.r1);    dst.Gth = T(src.Gth);
    dst.ke1  = T(src.ke1);  dst.ke2 = T(src.ke2);
    dst.ka1  = T(src.ka1);  dst.ka2 = T(src.ka2);   dst.kd  = T(src.kd);
    dst.Ts   = T(src.Ts);
    dst.BW   = T(src.BW);   dst.Gb  = T(src.Gb);    dst.Ib  = T(src.Ib);
  }

  template<typename T>
  void apply_adjustments(const T* theta, BasicParams<T>& p) const {
    for (int i = 0; i < n_params_; ++i) {
      const auto& desc = param_desc_[i];
      T multiplier = exp(theta[i]);

      if (desc.name == "kp1")  p.kp1  *= multiplier;
      if (desc.name == "kp2")  p.kp2  *= multiplier;
      if (desc.name == "kp3")  p.kp3  *= multiplier;
      if (desc.name == "Vm0")  p.Vm0  *= multiplier;
      if (desc.name == "Vmx")  p.Vmx  *= multiplier;
      if (desc.name == "p2U")  p.p2U  *= multiplier;
      if (desc.name == "ka1")  p.ka1  *= multiplier;
      if (desc.name == "ka2")  p.ka2  *= multiplier;
      if (desc.name == "kd")   p.kd   *= multiplier;
      if (desc.name == "kmax") p.kmax *= multiplier;
      if (desc.name == "kmin") p.kmin *= multiplier;
      if (desc.name == "kabs") p.kabs *= multiplier;
      if (desc.name == "kgri") p.kgri *= multiplier;
      if (desc.name == "VG")   p.VG   *= multiplier;
      if (desc.name == "VI")   p.VI   *= multiplier;
      if (desc.name == "Gb")   p.Gb   *= multiplier;
      if (desc.name == "Ib")   p.Ib   *= multiplier;
    }
  }

  template<typename T>
  static BasicParams<double> to_double(const BasicParams<T>& src) {
    BasicParams<double> dst;
    dst.VG   = extract_val(src.VG);   dst.k1  = extract_val(src.k1);   dst.k2  = extract_val(src.k2);
    dst.VI   = extract_val(src.VI);   dst.m1  = extract_val(src.m1);   dst.m2  = extract_val(src.m2);   dst.m4  = extract_val(src.m4);
    dst.kmax = extract_val(src.kmax); dst.kmin = extract_val(src.kmin); dst.kabs = extract_val(src.kabs); dst.kgri = extract_val(src.kgri);
    dst.f    = extract_val(src.f);    dst.b   = extract_val(src.b);     dst.d   = extract_val(src.d);
    dst.kp1  = extract_val(src.kp1);  dst.kp2 = extract_val(src.kp2);   dst.kp3 = extract_val(src.kp3);   dst.ki  = extract_val(src.ki);
    dst.Fcns = extract_val(src.Fcns); dst.Vm0 = extract_val(src.Vm0);   dst.Vmx = extract_val(src.Vmx);
    dst.Km0  = extract_val(src.Km0);  dst.p2U = extract_val(src.p2U);   dst.r1  = extract_val(src.r1);    dst.Gth = extract_val(src.Gth);
    dst.ke1  = extract_val(src.ke1);  dst.ke2 = extract_val(src.ke2);
    dst.ka1  = extract_val(src.ka1);  dst.ka2 = extract_val(src.ka2);   dst.kd  = extract_val(src.kd);
    dst.Ts   = extract_val(src.Ts);
    dst.BW   = extract_val(src.BW);   dst.Gb  = extract_val(src.Gb);    dst.Ib  = extract_val(src.Ib);
    return dst;
  }
};

// ---------------------------------------------------------------------------
// MAP prior — quadratic penalty (θᵢ − μᵢ) / σᵢ
// ---------------------------------------------------------------------------
struct PriorCost : ceres::CostFunction {
  std::vector<double> means_;
  std::vector<double> stds_;
  int n_params_;

  PriorCost(const std::vector<ParamDescriptor>& desc) {
    n_params_ = static_cast<int>(desc.size());
    set_num_residuals(n_params_);
    mutable_parameter_block_sizes()->push_back(n_params_);
    means_.resize(n_params_);
    stds_.resize(n_params_);
    for (int i = 0; i < n_params_; ++i) {
      means_[i] = desc[i].prior_mean;
      stds_[i] = desc[i].prior_std;
    }
  }

  bool Evaluate(const double* const* parameters, double* residuals,
                double** jacobians) const override {
    const double* params = parameters[0];
    for (int i = 0; i < n_params_; ++i) {
      residuals[i] = (params[i] - means_[i]) / stds_[i];
    }
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      for (int i = 0; i < n_params_; ++i) {
        for (int j = 0; j < n_params_; ++j)
          jacobians[0][i * n_params_ + j] = 0.0;
        jacobians[0][i * n_params_ + i] = 1.0 / stds_[i];
      }
    }
    return true;
  }

  double compute_scalar_prior(const double* params) const {
    double total = 0.0;
    for (int i = 0; i < n_params_; ++i) {
      double diff = (params[i] - means_[i]) / stds_[i];
      total += 0.5 * diff * diff;
    }
    return total;
  }
};

} // namespace uva_padova
