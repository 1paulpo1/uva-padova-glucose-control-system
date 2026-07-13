// =============================================================================
// FIM/covariance diagnostic test.
// Runs one identification trial and prints full FIM, covariance, and
// observation Jacobian diagnostics.
// =============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "uva_padova/types.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/integrals.hpp"
#include "uva_padova/delivery.hpp"
#include "uva_padova/cost_function.hpp"
#include "uva_padova/optimizer.hpp"
#include "uva_padova/diagnostics.hpp"
#include "uva_padova/constants.hpp"
#include <ceres/loss_function.h>

using namespace uva_padova;

static IdentificationData generate_data(const Params& true_p, int seed, bool with_bolus = false) {
  State ss;
  double insulin_ss = SteadyStateSolver::find(ss, true_p, true_p.Gb);

  struct Meal { int t; double cho_mg; };
  std::vector<Meal> meals = {
    { 60, 60000 }, { 360, 80000 }, { 600, 70000 },
  };

  struct Bolus { int t; double dose_pmol_per_kg; };
  std::vector<Bolus> boluses;
  if (with_bolus) {
    double bw = true_p.BW;
    boluses = {
      {60,  6.0 * constants::kInsulinUnits_To_Pmol / bw},
      {360, 8.0 * constants::kInsulinUnits_To_Pmol / bw},
      {600, 7.0 * constants::kInsulinUnits_To_Pmol / bw},
    };
  }

  State x = ss;
  BasicInputs<double> u;
  u.insulin_rate = insulin_ss;

  std::vector<double> cgm_time, cgm_value;
  std::vector<double> bolus_time_vec, bolus_dose_vec;
  std::mt19937 rng(seed);
  std::normal_distribution<double> noise(0.0, 3.0);

  int meal_idx = 0;
  int duration_min = 720;
  for (int t = 0; t <= duration_min; ++t) {
    if (meal_idx < (int)meals.size() && meals[meal_idx].t == t) {
      x.Qsto1 += meals[meal_idx].cho_mg;
      x.mealD = meals[meal_idx].cho_mg;
      meal_idx++;
    }
    for (auto& b : boluses) {
      if (b.t == t) {
        x.Isc1 += b.dose_pmol_per_kg;
        bolus_time_vec.push_back((double)b.t);
        bolus_dose_vec.push_back(b.dose_pmol_per_kg);
      }
    }
    if (t % 5 == 0) {
      cgm_time.push_back((double)t);
      cgm_value.push_back(x.Gsc + noise(rng));
    }
    if (t < duration_min)
      RK4Integrator::step(UvaPadovaODESystem{}, x, u, true_p, 1.0);
  }

  IdentificationData data;
  data.cgm_time  = cgm_time;
  data.cgm_value = cgm_value;
  data.bolus_time  = bolus_time_vec;
  data.bolus_dose  = bolus_dose_vec;
  for (auto& m : meals) {
    data.meal_time.push_back((double)m.t);
    data.meal_cho.push_back(m.cho_mg);
  }
  data.finalize();
  return data;
}

int main() {
  std::cout << std::scientific << std::setprecision(6);

  Params true_p = Params::defaults_t1d();
  true_p.kp2 = 0.0030;
  true_p.Vm0 = 3.20;
  true_p.p2U = 0.0280;

  std::vector<ParamDescriptor> desc = {
    { "kp2", 0.0, 1.0, 0.0 },
    { "Vm0", 0.0, 1.0, 0.0 },
    { "p2U", 0.0, 1.0, 0.0 },
  };

  auto data = generate_data(true_p, 42, true);
  auto result = IdentificationPipeline<>::run(data, Params::defaults_t1d(), desc, 3, 500);

  // Print result cov
  std::cout << "\n=== Pipeline result ===\n";
  std::cout << "  cost=" << result.cost << "\n";
  std::cout << "  theta = [";
  for (size_t i = 0; i < result.params.size(); ++i)
    std::cout << (i ? ", " : "") << result.params[i];
  std::cout << "]\n";

  auto recovered = result.to_params(Params::defaults_t1d(), desc);
  double rk = std::abs(recovered.kp2 - true_p.kp2) / true_p.kp2;
  double rv = std::abs(recovered.Vm0 - true_p.Vm0) / true_p.Vm0;
  double rp = std::abs(recovered.p2U - true_p.p2U) / true_p.p2U;
  std::cout << "  rel_err: kp2=" << rk << " Vm0=" << rv << " p2U=" << rp << "\n";

  std::cout << "\n=== Posterior cov (from pipeline) ===\n";
  if (result.cov.rows() > 0) {
    std::cout << result.cov << "\n";
    double c12 = result.cov(0,1) / std::sqrt(result.cov(0,0) * result.cov(1,1));
    double c13 = result.cov(0,2) / std::sqrt(result.cov(0,0) * result.cov(2,2));
    double c23 = result.cov(1,2) / std::sqrt(result.cov(1,1) * result.cov(2,2));
    std::cout << "  corr(kp2,Vm0)=" << c12 << "\n";
    std::cout << "  corr(kp2,p2U)=" << c13 << "\n";
    std::cout << "  corr(Vm0,p2U)=" << c23 << "\n";
  } else {
    std::cout << "  (empty)\n";
  }

  // ---- Gate (a): multiplicity of minima ----
  {
    auto gate_a = diagnose_multiplicity(result.all_restarts, desc);
    std::cout << "\n=== Gate (a): multiplicity of minima ===\n";
    std::cout << "  restarts=" << result.all_restarts.size()
              << " clusters=" << gate_a.n_clusters
              << " labels=[";
    for (size_t i = 0; i < gate_a.labels.size(); ++i)
      std::cout << (i ? " " : "") << gate_a.labels[i];
    std::cout << "] passed=" << (gate_a.passed ? "yes" : "NO") << "\n";
    if (gate_a.passed) {
      std::cout << "  ✓ Single basin of attraction\n";
    } else {
      std::cout << "  ✗ Multiple basins → personalisation BLOCKED\n";
    }

    // Write restart data for visualisation
    std::ofstream csv("uva-padova-cpp/validation/output/gate_restarts.csv");
    csv << "type,cluster";
    for (auto& d : desc) csv << "," << d.name;
    csv << "\n";
    for (size_t r = 0; r < result.all_restarts.size(); ++r) {
      csv << "restart," << gate_a.labels[r];
      for (double v : result.all_restarts[r].params) csv << "," << v;
      csv << "\n";
    }
    csv << "estimate,-1";
    for (double v : result.params) csv << "," << v;
    csv << "\n";
    csv << "truth,-1,"
        << true_p.kp2 << "," << true_p.Vm0 << "," << true_p.p2U << "\n";
  }

  // ---- Gate (b): structural adequacy ----
  {
    Params base_p = Params::defaults_t1d();
    ObservationCost<> obs_cost(data, base_p, desc);
    PriorCost prior_cost(desc);

    auto gate_b = diagnose_structural_adequacy(
        data.cgm_time, data.cgm_value, data.meal_time,
        result.params.data(), obs_cost);

    std::cout << "\n=== Gate (b): structural adequacy ===\n";
    std::cout << "  Ljung-Box: Q=" << gate_b.ljung_box.Q
              << " p=" << gate_b.ljung_box.p_value
              << " lags=" << gate_b.ljung_box.n_lags
              << " significant=" << (gate_b.ljung_box.significant ? "yes" : "no") << "\n";
    for (auto& rs : gate_b.regimes) {
      std::cout << "    " << rs.name
                << ": n=" << rs.count
                << " bias=" << rs.bias_mgdl << " mg/dL"
                << " mean_glucose=" << rs.mean_glucose << "\n";
    }
    std::cout << "  Regime bias detected: " << (gate_b.regime_bias_detected ? "yes" : "no") << "\n";
    if (gate_b.passed) {
      std::cout << "  ✓ No structural issues detected\n";
    } else {
      std::cout << "  " << gate_b.description << "\n";
    }
  }

  // ---- Gate (c): observability (SVD/FIM) ----
  {
    Params base_p = Params::defaults_t1d();
    ObservationCost<> obs_cost(data, base_p, desc);

    auto gate_c = diagnose_observability(
        result.params.data(), obs_cost, desc);

    std::cout << "\n=== Gate (c): observability (SVD/FIM) ===\n";
    std::cout << "  Eigenvalues:";
    for (int i = 0; i < gate_c.n_params; ++i)
      std::cout << " " << gate_c.eigenvalues[i];
    std::cout << "\n";
    std::cout << "  Condition number: " << gate_c.condition_number << "\n";
    std::cout << "  Weak threshold (" << 1e-3 << " of max): "
              << 1e-3 * gate_c.max_eigenvalue << "\n";
    std::cout << "  Per-param labels:";
    for (auto& lbl : gate_c.param_labels)
      std::cout << " " << lbl;
    std::cout << "\n";
  }

  // Now rebuild the Jacobian ourselves for diagnostics
  std::cout << "\n=== Direct Jacobian evaluation at final params ===\n";
  Params base_p = Params::defaults_t1d();
  ObservationCost<> obs_cost(data, base_p, desc);
  PriorCost prior_cost(desc);

  int n_obs = obs_cost.n_obs_;
  int n_params = obs_cost.n_params_;

  const double* theta = result.params.data();

  // Observation Jacobian
  std::vector<double> obs_resid(n_obs);
  std::vector<double> obs_J_buf(n_obs * n_params);
  double* obs_J_ptrs[] = { obs_J_buf.data() };
  obs_cost.Evaluate(&theta, obs_resid.data(), obs_J_ptrs);

  Eigen::MatrixXd J_obs(n_obs, n_params);
  for (int i = 0; i < n_obs; ++i)
    for (int j = 0; j < n_params; ++j)
      J_obs(i, j) = obs_J_buf[i * n_params + j];

  std::cout << "  J_obs shape: " << J_obs.rows() << " x " << J_obs.cols() << "\n";
  std::cout << "  J_obs column norms: "
            << J_obs.col(0).norm() << "  "
            << J_obs.col(1).norm() << "  "
            << J_obs.col(2).norm() << "\n";

  // FIM = J^T J
  Eigen::MatrixXd FIM_obs = J_obs.transpose() * J_obs;
  std::cout << "\n  FIM_obs = J_obs^T J_obs:\n" << FIM_obs << "\n";

  // Condition number of FIM_obs
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(FIM_obs);
  std::cout << "  FIM_obs singular values: " << svd.singularValues().transpose() << "\n";
  // NOTE: these singular values come from the sqrt-trick Jacobian (Evaluate).
  // Gate (c) prints "Eigenvalues" from the raw normalised Jacobian
  // (compute_raw_gradients) — they differ by construction because the
  // residual transformation is different.  This is not a bug.
  std::cout << "  FIM_obs cond: " << svd.singularValues()(0)
            << " / " << svd.singularValues()(n_params-1)
            << " = " << (svd.singularValues()(0) / svd.singularValues()(n_params-1)) << "\n";

  // Prior Jacobian
  std::vector<double> prior_resid(n_params);
  std::vector<double> prior_J_buf(n_params * n_params);
  double* prior_J_ptrs[] = { prior_J_buf.data() };
  prior_cost.Evaluate(&theta, prior_resid.data(), prior_J_ptrs);

  Eigen::MatrixXd J_prior(n_params, n_params);
  for (int i = 0; i < n_params; ++i)
    for (int j = 0; j < n_params; ++j)
      J_prior(i, j) = prior_J_buf[i * n_params + j];

  std::cout << "\n  J_prior:\n" << J_prior << "\n";
  Eigen::MatrixXd FIM_prior = J_prior.transpose() * J_prior;
  std::cout << "  FIM_prior:\n" << FIM_prior << "\n";

  Eigen::MatrixXd FIM_total = FIM_obs + FIM_prior;
  std::cout << "\n  FIM_total = FIM_obs + FIM_prior:\n" << FIM_total << "\n";

  // Posterior cov from direct computation
  constexpr double kReg = 1e-12;
  Eigen::MatrixXd cov_direct = (FIM_total + kReg * Eigen::MatrixXd::Identity(n_params, n_params)).inverse();
  std::cout << "\n  cov_direct = (FIM_total + reg*I)^-1:\n" << cov_direct << "\n";

  // Correlations
  auto print_corr = [](const Eigen::MatrixXd& c) {
    for (int i = 0; i < c.rows(); ++i)
      for (int j = i+1; j < c.cols(); ++j)
        std::cout << "    corr(" << i << "," << j << ")="
                  << c(i,j)/std::sqrt(c(i,i)*c(j,j)) << "\n";
  };
  std::cout << "  Correlations from cov_direct:\n";
  print_corr(cov_direct);

  // Now check: what happens if we DON'T use the prior?
  std::cout << "\n=== WITHOUT prior (observation-only) ===\n";
  Eigen::MatrixXd cov_obs_only = (FIM_obs + kReg * Eigen::MatrixXd::Identity(n_params, n_params)).inverse();
  std::cout << "  cov_obs_only = (FIM_obs + reg*I)^-1:\n" << cov_obs_only << "\n";
  std::cout << "  Correlations (obs only):\n";
  print_corr(cov_obs_only);

  // Check if J_obs columns are correlated by computing the Gram matrix
  Eigen::MatrixXd gram(n_params, n_params);
  for (int i = 0; i < n_params; ++i)
    for (int j = 0; j < n_params; ++j)
      gram(i,j) = J_obs.col(i).dot(J_obs.col(j)) / (J_obs.col(i).norm() * J_obs.col(j).norm() + 1e-30);
  std::cout << "\n  Column angle cosines (Gram matrix):\n" << gram << "\n";

  // ======================================================================
  // J_obs at θ=0 (true params): compare no_bolus vs bolus at FIXED point
  // ======================================================================
  std::cout << "\n=== J_obs column norms at θ=0 (true params) ===\n";

  auto compare_jobs_at_zero = [&](bool with_bolus) {
    auto d = generate_data(true_p, 42, with_bolus);
    ObservationCost<> oc(d, base_p, desc);
    int no = oc.n_obs_, np = oc.n_params_;
    std::vector<double> zero_theta(np, 0.0);
    const double* zt = zero_theta.data();
    std::vector<double> buf(no * np);
    double* ptrs[] = { buf.data() };
    std::vector<double> resid(no);
    oc.Evaluate(&zt, resid.data(), ptrs);
    Eigen::MatrixXd J(no, np);
    for (int i = 0; i < no; ++i)
      for (int j = 0; j < np; ++j)
        J(i, j) = buf[i * np + j];
    return J;
  };

  Eigen::MatrixXd J0_nb = compare_jobs_at_zero(false);
  Eigen::MatrixXd J0_wb = compare_jobs_at_zero(true);
  auto print_col_norms = [&](const Eigen::MatrixXd& J, const char* label) {
    std::cout << "  " << label << ": "
              << J.col(0).norm() << "  "
              << J.col(1).norm() << "  "
              << J.col(2).norm() << "\n";
  };
  print_col_norms(J0_nb, "no_bolus  ");
  print_col_norms(J0_wb, "with_bolus");

  // ======================================================================
  // Sandwich covariance: H⁻¹ V H⁻¹ with raw residuals (no sqrt-trick)
  // ======================================================================
  std::cout << "\n=== Sandwich covariance ===\n";

  // Log theta for comparison with pipeline
  std::cout << "  [debug sandwich] theta = ["
            << theta[0] << " " << theta[1] << " " << theta[2] << "]\n";

  // Get raw residuals (r_i, not sqrt-transformed) and their Jacobian
  std::vector<double> raw_resid;
  Eigen::MatrixXd J_raw;
  bool ok = obs_cost.compute_raw_gradients(theta, raw_resid, J_raw);
  if (!ok) {
    std::cout << "  [debug sandwich] compute_raw_gradients failed (NaN)\n";
    return 0;
  }

  Eigen::MatrixXd Hs = Eigen::MatrixXd::Zero(n_params, n_params);
  Eigen::MatrixXd Vs = Eigen::MatrixXd::Zero(n_params, n_params);

  // NOTE: uses per-observation loss as a robust working model for the
  // sandwich estimator. This differs from the aggregate SoftLOneLoss used
  // in the optimization objective (compute_scalar_cost and Ceres), which
  // is intentional — the sandwich estimator is designed to be robust to
  // working model misspecification, and per-observation loss yields a
  // full-rank V matrix (aggregate loss would give rank-1 V).
  ceres::SoftLOneLoss loss(1.0);
  double rho[3];

  for (int i = 0; i < n_obs; ++i) {
    double r   = raw_resid[i];
    double s   = r * r;
    loss.Evaluate(s, rho);
    double drho_dr   = rho[1] * 2.0 * r;
    double d2rho_dr2 = rho[1] * 2.0 + rho[2] * 4.0 * s;
    auto g     = J_raw.row(i);
    Hs += d2rho_dr2 * g.transpose() * g;
    Vs += (drho_dr * drho_dr) * g.transpose() * g;
  }

  // Prior Hessian: Σ₀⁻¹ = diag(1/σ_i²)
  for (int i = 0; i < n_params; ++i) {
    double iv = 1.0 / (prior_cost.stds_[i] * prior_cost.stds_[i]);
    Hs(i, i) += iv;
  }

  std::cout << "\n  H (Σ ρ''·∇r·∇rᵀ + Σ₀⁻¹):\n" << Hs << "\n";
  std::cout << "  V (Σ ρ'²·∇r·∇rᵀ):\n" << Vs << "\n";

  constexpr double kRegS = 1e-12;
  Eigen::MatrixXd Hs_inv = (Hs + kRegS * Eigen::MatrixXd::Identity(n_params, n_params)).inverse();
  Eigen::MatrixXd cov_sandwich_direct = Hs_inv * Vs * Hs_inv;

  std::cout << "\n  cov_sandwich = H⁻¹·V·H⁻¹:\n" << cov_sandwich_direct << "\n";

  // Compare with pipeline result
  if (result.cov_sandwich.rows() > 0) {
    std::cout << "\n  cov_sandwich (from pipeline):\n" << result.cov_sandwich << "\n";
    double diff = (cov_sandwich_direct - result.cov_sandwich).norm();
    std::cout << "  Frobenius diff: " << diff << "\n";
  }

  // Ratio sandwich / naive standard deviations
  std::cout << "\n  Std ratio sandwich/naive:\n";
  for (int i = 0; i < n_params; ++i) {
    double naive_std  = std::sqrt(cov_direct(i, i));
    double sand_std   = std::sqrt(cov_sandwich_direct(i, i));
    std::cout << "    θ[" << i << "]: " << sand_std << " / " << naive_std
              << " = " << (sand_std / naive_std) << "\n";
  }

  // Correlations from sandwich
  std::cout << "\n  Correlations from sandwich:\n";
  auto print_sand_corr = [](const Eigen::MatrixXd& c) {
    for (int i = 0; i < c.rows(); ++i)
      for (int j = i+1; j < c.cols(); ++j)
        std::cout << "    corr(" << i << "," << j << ")="
                  << c(i,j)/std::sqrt(c(i,i)*c(j,j)) << "\n";
  };
  print_sand_corr(cov_sandwich_direct);

  return 0;
}
