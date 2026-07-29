#pragma once

#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <numeric>

#include <Eigen/Dense>
#include <ceres/jet.h>

#include <ceres/cost_function.h>

#include "uva_padova/types.hpp"
#include "uva_padova/simulator.hpp"

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
  bool fixed{false};
  bool online_adaptive{false};
};

// ---------------------------------------------------------------------------
// Observation cost — returns plain standardized residuals for LM.
// Robustness is handled by ceres::SoftLOneLoss passed in AddResidualBlock.
//
//   r = (G_sim − G_obs) / σ
//   residual = r
//
// Ceres minimizes Σ 2·(√(1+r²) − 1) via SoftLOneLoss.
//
// Parameter vector θ lives in LOG-space:
//   true_value = base_value · exp(θᵢ)
// ---------------------------------------------------------------------------
template <typename Simulator = UvaPadovaSimulator>
struct ObservationCost : ceres::CostFunction {
  static constexpr int kMaxJetDim = 30;
  using JetT = ceres::Jet<double, kMaxJetDim>;

  const IdentificationData data_;
  const Params base_params_;
  const std::vector<ParamDescriptor> param_desc_;
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

  ObservationCost(const IdentificationData& data,
                  const Params& base_params,
                  const std::vector<ParamDescriptor>& param_desc,
                  double sigma)
    : data_(data), base_params_(base_params),
      param_desc_(param_desc), n_params_(static_cast<int>(param_desc.size())),
      n_obs_(static_cast<int>(data.cgm_time.size())), sigma_(sigma) {
    if (n_params_ > kMaxJetDim) {
      std::cerr << "ObservationCost: n_params=" << n_params_
                << " exceeds kMaxJetDim=" << kMaxJetDim << "\n";
      std::abort();
    }
    set_num_residuals(n_obs_);
    mutable_parameter_block_sizes()->push_back(n_params_);
  }

  // ---- LM interface (plain residuals, SoftLOneLoss via Ceres) ----
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
        residuals[i] = extract_val(r);
        if (!std::isfinite(residuals[i])) return false;
        for (int j = 0; j < n_params_; ++j) {
          double d = extract_deriv(r, j);
          if (!std::isfinite(d)) return false;
          jacobians[0][i * n_params_ + j] = d;
        }
      }
    } else {
      auto cgm = simulate_observations<double>(params);
      for (int i = 0; i < n_obs_; ++i) {
        double r = (cgm[i] - data_.cgm_value[i]) / sigma_;
        residuals[i] = r;
        if (!std::isfinite(residuals[i])) return false;
      }
    }
    return true;
  }

  // ---- Raw residuals + Jacobian (retained for reference/diagnostics) ----
  // Returns r_i = (G_sim - G_obs) / σ and J_raw[i,j] = ∂r_i/∂θ_j.
  // NOTE: with plain residuals in Evaluate(), this is now equivalent
  // to Evaluate() — kept for backward compatibility and diagnostics (Gate C).
  bool compute_raw_gradients(const double* params,
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
      if (!std::isfinite(raw_residuals[i])) return false;
      for (int j = 0; j < n_params_; ++j) {
        double d = extract_deriv(r, j);
        if (!std::isfinite(d)) return false;
        J_raw(i, j) = d;
      }
    }
    return true;
  }

  // ---- Scalar cost (matches Ceres SoftLOneLoss on aggregate ||r||²) ----
  // Returns √(1 + Σᵢ rᵢ²) − 1  — matches Ceres ½·SoftLOneLoss(||r||²).
  double compute_scalar_cost(const double* params) const {
    eval_count_++;
    auto cgm = simulate_observations<double>(params);
    double sum_sq = 0.0;
    for (int i = 0; i < n_obs_; ++i) {
      double r = (cgm[i] - data_.cgm_value[i]) / sigma_;
      if (!std::isfinite(r)) return 1e20;
      sum_sq += r * r;
    }
    return std::sqrt(1.0 + sum_sq) - 1.0;
  }

  // ---- Helpers for NaN-derivative detection and salvage ----
  template<typename T>
  static bool scalar_is_finite(const T& x) {
    if constexpr (std::is_same_v<T, double>) {
      return std::isfinite(x);
    } else {
      if (!std::isfinite(x.a)) return false;
      for (int i = 0; i < x.v.size(); ++i)
        if (!std::isfinite(x.v[i])) return false;
      return true;
    }
  }

  template<typename T>
  static bool state_is_finite(const BasicState<T>& s) {
    return scalar_is_finite(s.Gp) && scalar_is_finite(s.Gt) &&
           scalar_is_finite(s.Qsto1) && scalar_is_finite(s.Qsto2) &&
           scalar_is_finite(s.Qgut) && scalar_is_finite(s.Ip) &&
           scalar_is_finite(s.Ii) && scalar_is_finite(s.Isc1) &&
           scalar_is_finite(s.Isc2) && scalar_is_finite(s.I_prime) &&
           scalar_is_finite(s.Xl) && scalar_is_finite(s.X) &&
           scalar_is_finite(s.Gsc) && scalar_is_finite(s.mealD);
  }

  template<typename T>
  static void salvage_state(BasicState<T>& s) {
    if constexpr (!std::is_same_v<T, double>) {
      auto fix = [](T& v) {
        if (!std::isfinite(v.a)) { v.a = 0.0; v.v.setZero(); return; }
        for (int i = 0; i < v.v.size(); ++i)
          if (!std::isfinite(v.v[i])) v.v[i] = 0.0;
      };
      fix(s.Gp); fix(s.Gt); fix(s.Qsto1); fix(s.Qsto2); fix(s.Qgut);
      fix(s.Ip); fix(s.Ii); fix(s.Isc1); fix(s.Isc2); fix(s.I_prime);
      fix(s.Xl); fix(s.X); fix(s.Gsc); fix(s.mealD);
    }
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
    double insulin_ss = Simulator::find_steady_state(ss_state, p_double, p_double.Gb);

    // Convert SS to T with sensitivities
    BasicState<T> state;
    T basal_rate(0);

    if constexpr (std::is_same_v<T, double>) {
      for (int i = 0; i < BasicState<T>::kSize; ++i)
        state.at(i) = T(ss_state.at(i));
      basal_rate = T(insulin_ss);
    } else {
      Eigen::Matrix<double, kMaxJetDim, 1> di_ss;
      di_ss.setZero();
      using AlignedMat = Eigen::Matrix<double, kMaxJetDim, 1>;
      std::vector<AlignedMat, Eigen::aligned_allocator<AlignedMat>> dstate(BasicState<T>::kSize);
      for (auto& v : dstate) v.setZero();

      // --- Implicit Function Theorem: differentiate SS system w.r.t. θ ---
      // At the converged SS: F(x_ss, u_basal, p(θ)) = 0  (14 ODE equations)
      //                      h(x_ss, p(θ)) = Gp/VG - Gb  (glucose target)
      // Differentiating:
      //   [∂F/∂x  ∂F/∂u] [dx/dθ]   = -[∂F/∂θ]
      //   [∂h/∂x   0   ] [du/dθ]     -[∂h/∂θ]
      // One Jet<30> RHS evaluation gives all partials simultaneously.
      // Note: kMaxJetDim=30 is used as the Jet dimension; the state (14),
      // u_basal (1), and active params (≤ 9) fit with capacity to spare.
      constexpr int kStateDim = BasicState<T>::kSize;  // 14
      using JetAll = ceres::Jet<double, kMaxJetDim>;

      BasicState<JetAll> ss_jet;
      for (int j = 0; j < kStateDim; ++j)
        ss_jet.at(j) = JetAll(ss_state.at(j), j);

      BasicInputs<JetAll> u_jet;
      u_jet.insulin_rate = JetAll(insulin_ss, kStateDim);

      BasicParams<JetAll> p_jet;
      copy_params(base_params_, p_jet);
      std::vector<JetAll> theta_jet(n_params_);
      for (int pi = 0; pi < n_params_; ++pi)
        theta_jet[pi] = JetAll(extract_val(params[pi]), kStateDim + 1 + pi);
      apply_adjustments(theta_jet.data(), p_jet);

      auto rhs_all = UvaPadovaODESystem::rhs(ss_jet, u_jet, p_jet);

      Eigen::Matrix<double, kStateDim, kStateDim + 1> J_xu;
      Eigen::Matrix<double, kStateDim, Eigen::Dynamic> J_theta(kStateDim, n_params_);
      for (int j = 0; j < kStateDim; ++j) {
        auto& dx_j = rhs_all.at(j);
        for (int k = 0; k < kStateDim; ++k)
          J_xu(j, k) = dx_j.v[k];
        J_xu(j, kStateDim) = dx_j.v[kStateDim];
        for (int pi = 0; pi < n_params_; ++pi)
          J_theta(j, pi) = dx_j.v[kStateDim + 1 + pi];
      }

      Eigen::Matrix<double, 1, kStateDim> dh_dx;
      dh_dx.setZero();
      dh_dx(0) = 1.0 / p_double.VG;

      Eigen::Matrix<double, 1, Eigen::Dynamic> dh_dtheta(1, n_params_);
      dh_dtheta.setZero();
      for (int pi = 0; pi < n_params_; ++pi)
        if (param_desc_[pi].name == "Gb")
          dh_dtheta(pi) = -p_double.Gb;

      Eigen::Matrix<double, kStateDim + 1, kStateDim + 1> M;
      M.topLeftCorner(kStateDim, kStateDim + 1) = J_xu;
      M.template bottomRows<1>().leftCols(kStateDim) = dh_dx;
      M(kStateDim, kStateDim) = 0.0;

      Eigen::Matrix<double, kStateDim + 1, Eigen::Dynamic> RHS(kStateDim + 1, n_params_);
      RHS.topRows(kStateDim) = -J_theta;
      RHS.template bottomRows<1>() = -dh_dtheta;

      bool implicit_ok = true;
      Eigen::Matrix<double, kStateDim + 1, Eigen::Dynamic> S(kStateDim + 1, n_params_);
      {
        auto qr = M.colPivHouseholderQr();
        qr.setThreshold(1e-30);
        if (qr.rank() < kStateDim + 1) {
          implicit_ok = false;
        } else {
          S = qr.solve(RHS);
          if (!S.allFinite())
            implicit_ok = false;
        }
      }

      if (implicit_ok) {
        for (int pi = 0; pi < n_params_; ++pi) {
          di_ss[pi] = S(kStateDim, pi);
          for (int j = 0; j < kStateDim; ++j)
            dstate[j][pi] = S(j, pi);
        }
      } else {
      // Fallback: finite differences (if implicit approach failed or dim mismatch)
        const double eps = 1e-6;
        double theta_double[kMaxJetDim];
        for (int i = 0; i < n_params_; ++i)
          theta_double[i] = extract_val(params[i]);

        for (int pi = 0; pi < n_params_; ++pi) {
          double saved = theta_double[pi];
          theta_double[pi] += eps;

          BasicParams<double> p_pert;
          copy_params(base_params_, p_pert);
          apply_adjustments(theta_double, p_pert);

          State ss_pert;
          double i_ss_pert = Simulator::find_steady_state(ss_pert, p_pert, p_pert.Gb);

          double inv_eps = 1.0 / eps;
          for (int j = 0; j < BasicState<T>::kSize; ++j)
            dstate[j][pi] = (ss_pert.at(j) - ss_state.at(j)) * inv_eps;
          di_ss[pi] = (i_ss_pert - insulin_ss) * inv_eps;
          theta_double[pi] = saved;
        }
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
          Simulator::template notify<0>(state, u, T(t));
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
          Simulator::template notify<1>(state, u, T(t));
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
          Simulator::template notify<2>(state, u, T(t));
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
      if (t < data_.total_min()) {
        Simulator::step(state, u, p, 1.0);
        salvage_state(state);
      }
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
    dst.VI   = T(src.VI);   dst.m1  = T(src.m1);   dst.m2  = T(src.m2);   dst.m4  = T(src.m4);   dst.HE_b = T(src.HE_b);
    dst.kmax = T(src.kmax); dst.kmin = T(src.kmin); dst.kabs = T(src.kabs); dst.kgri = T(src.kgri);
    dst.f    = T(src.f);    dst.b   = T(src.b);     dst.d   = T(src.d);
    dst.kp1  = T(src.kp1);  dst.kp2 = T(src.kp2);   dst.kp3 = T(src.kp3);   dst.ki  = T(src.ki);
    dst.Fcns = T(src.Fcns); dst.Vm0 = T(src.Vm0);   dst.Vmx = T(src.Vmx);
    dst.Km0  = T(src.Km0);  dst.p2U = T(src.p2U);   dst.r1  = T(src.r1);    dst.r2  = T(src.r2);    dst.Gth = T(src.Gth);
    dst.ke1  = T(src.ke1);  dst.ke2 = T(src.ke2);
    dst.ka1  = T(src.ka1);  dst.ka2 = T(src.ka2);   dst.kd  = T(src.kd);
    dst.Ts   = T(src.Ts);
    dst.BW   = T(src.BW);   dst.Gb  = T(src.Gb);    dst.Ib  = T(src.Ib);
  }

  template<typename T>
  void apply_adjustments(const T* theta, BasicParams<T>& p) const {
    for (int i = 0; i < n_params_; ++i) {
      T multiplier = exp(theta[i]);
      if (!scalar_is_finite(multiplier)) {
        if constexpr (std::is_same_v<T, double>) {
          multiplier = T(1.0);
        } else {
          multiplier.a = 1.0;
          multiplier.v.setZero();
        }
      }
      param_by_id(p, param_id_from_string(param_desc_[i].name)) *= multiplier;
    }
  }

  template<typename T>
  static BasicParams<double> to_double(const BasicParams<T>& src) {
    BasicParams<double> dst;
    dst.VG   = extract_val(src.VG);   dst.k1  = extract_val(src.k1);   dst.k2  = extract_val(src.k2);
    dst.VI   = extract_val(src.VI);   dst.m1  = extract_val(src.m1);   dst.m2  = extract_val(src.m2);   dst.m4  = extract_val(src.m4);   dst.HE_b = extract_val(src.HE_b);
    dst.kmax = extract_val(src.kmax); dst.kmin = extract_val(src.kmin); dst.kabs = extract_val(src.kabs); dst.kgri = extract_val(src.kgri);
    dst.f    = extract_val(src.f);    dst.b   = extract_val(src.b);     dst.d   = extract_val(src.d);
    dst.kp1  = extract_val(src.kp1);  dst.kp2 = extract_val(src.kp2);   dst.kp3 = extract_val(src.kp3);   dst.ki  = extract_val(src.ki);
    dst.Fcns = extract_val(src.Fcns); dst.Vm0 = extract_val(src.Vm0);   dst.Vmx = extract_val(src.Vmx);
    dst.Km0  = extract_val(src.Km0);  dst.p2U = extract_val(src.p2U);   dst.r1  = extract_val(src.r1);    dst.r2  = extract_val(src.r2);    dst.Gth = extract_val(src.Gth);
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
