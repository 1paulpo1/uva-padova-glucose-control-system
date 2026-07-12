#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "virtual_patient.hpp"
#include "uva_padova/config_loader.hpp"
#include "uva_padova/optimizer.hpp"
#include "uva_padova/diagnostics.hpp"
#include "uva_padova/cost_function.hpp"

using namespace uva_padova;
using namespace uva_padova::crossval;
using json = nlohmann::json;

// ===========================================================================
// RAII guard: suppress std::cout during identification
// ===========================================================================
struct CoutSilencer {
  std::streambuf* old;
  std::ofstream null;
  CoutSilencer() : old(std::cout.rdbuf()), null("/dev/null") {
    std::cout.rdbuf(null.rdbuf());
  }
  ~CoutSilencer() { std::cout.rdbuf(old); }
};

// ===========================================================================
// Extract parameter value from Params by name
// ===========================================================================
static double get_param_value(const Params& p, const std::string& name) {
  if (name == "VG")   return p.VG;
  if (name == "k1")   return p.k1;
  if (name == "k2")   return p.k2;
  if (name == "VI")   return p.VI;
  if (name == "m1")   return p.m1;
  if (name == "m2")   return p.m2;
  if (name == "m4")   return p.m4;
  if (name == "kmax") return p.kmax;
  if (name == "kmin") return p.kmin;
  if (name == "kabs") return p.kabs;
  if (name == "kgri") return p.kgri;
  if (name == "f")    return p.f;
  if (name == "b")    return p.b;
  if (name == "d")    return p.d;
  if (name == "kp1")  return p.kp1;
  if (name == "kp2")  return p.kp2;
  if (name == "kp3")  return p.kp3;
  if (name == "ki")   return p.ki;
  if (name == "Fcns") return p.Fcns;
  if (name == "Vm0")  return p.Vm0;
  if (name == "Vmx")  return p.Vmx;
  if (name == "Km0")  return p.Km0;
  if (name == "p2U")  return p.p2U;
  if (name == "r1")   return p.r1;
  if (name == "Gth")  return p.Gth;
  if (name == "ke1")  return p.ke1;
  if (name == "ke2")  return p.ke2;
  if (name == "ka1")  return p.ka1;
  if (name == "ka2")  return p.ka2;
  if (name == "kd")   return p.kd;
  if (name == "Ts")   return p.Ts;
  if (name == "BW")   return p.BW;
  if (name == "Gb")   return p.Gb;
  if (name == "Ib")   return p.Ib;
  return 0.0;
}

// ===========================================================================
// Filter zero-valued CGM observations (dropout convention: 0 = missing)
// ===========================================================================
static IdentificationData filter_cgm_zeros(const IdentificationData& src) {
  IdentificationData out = src;
  out.cgm_time.clear();
  out.cgm_value.clear();
  out.meal_time = src.meal_time;
  out.meal_cho = src.meal_cho;
  out.insulin_time = src.insulin_time;
  out.insulin_rate = src.insulin_rate;
  out.bolus_time = src.bolus_time;
  out.bolus_dose = src.bolus_dose;
  for (size_t i = 0; i < src.cgm_time.size(); ++i) {
    if (src.cgm_value[i] > 1e-6) {
      out.cgm_time.push_back(src.cgm_time[i]);
      out.cgm_value.push_back(src.cgm_value[i]);
    }
  }
  out.finalize();
  return out;
}

// ===========================================================================
// CSV writing
// ===========================================================================
static void write_csv_header(std::ofstream& csv, const std::vector<ParamDescriptor>& desc) {
  csv << "patient_id,success,cost,total_evals,"
      << "gateA_passed,gateA_n_clusters,"
      << "gateB_passed,gateB_lb_pval,gateB_regime_bias,"
      << "gateC_n_dominated";
  for (const auto& d : desc) {
    csv << ",t_" << d.name << ",r_" << d.name << ",e_" << d.name << ",e_phys_" << d.name;
  }
  csv << "\n";
}

static void write_csv_row(std::ofstream& csv, int pid, const IdentificationPipeline<>::Result& ires,
                          const VirtualPatientResult& vp,
                          const GateAResult& ga, const GateBResult& gb, const GateCResult& gc,
                          const std::vector<double>& abs_err,
                          const std::vector<double>& abs_err_phys)
{
  csv << pid << ","
      << (ires.success ? 1 : 0) << ","
      << ires.cost << ","
      << ires.total_evals << ","
      << (ga.passed ? 1 : 0) << ","
      << ga.n_clusters << ","
      << (gb.passed ? 1 : 0) << ","
      << gb.ljung_box.p_value << ","
      << (gb.regime_bias_detected ? 1 : 0) << ","
      << static_cast<int>(std::count(gc.prior_dominated.begin(), gc.prior_dominated.end(), true));

  for (size_t i = 0; i < vp.active_desc.size(); ++i) {
    double t = vp.theta_true[i];
    double r = ires.success ? ires.params[i] : std::numeric_limits<double>::quiet_NaN();
    double e = ires.success ? abs_err[i] : std::numeric_limits<double>::quiet_NaN();
    double ep = ires.success ? abs_err_phys[i] : std::numeric_limits<double>::quiet_NaN();
    csv << "," << t << "," << r << "," << e << "," << ep;
  }
  csv << "\n";
}

// ===========================================================================
// JSON building
// ===========================================================================
static json build_json_result(int pid, const IdentificationPipeline<>::Result& ires,
                              const VirtualPatientResult& vp,
                              const GateAResult& ga, const GateBResult& gb, const GateCResult& gc,
                              const std::vector<double>& abs_err,
                              const std::vector<double>& abs_err_phys,
                              double elapsed_s)
{
  json j;
  j["patient_id"] = pid;
  j["success"] = ires.success;
  j["cost"] = ires.cost;
  j["total_evals"] = ires.total_evals;
  j["elapsed_s"] = elapsed_s;

  j["gate_a"]["passed"] = ga.passed;
  j["gate_a"]["n_clusters"] = ga.n_clusters;

  j["gate_b"]["passed"] = gb.passed;
  j["gate_b"]["lb_q"] = gb.ljung_box.Q;
  j["gate_b"]["lb_p_value"] = gb.ljung_box.p_value;
  j["gate_b"]["lb_significant"] = gb.ljung_box.significant;
  j["gate_b"]["regime_bias"] = gb.regime_bias_detected;
  j["gate_b"]["description"] = gb.description;

  j["gate_c"]["n_prior_dominated"] = static_cast<int>(
      std::count(gc.prior_dominated.begin(), gc.prior_dominated.end(), true));

  json theta_true = json::array();
  json theta_rec = json::array();
  json err = json::array();
  json err_phys = json::array();
  json rec_phys = json::array();
  json true_phys = json::array();
  Params recovered;
  if (ires.success)
    recovered = ires.to_params(vp.base_params, vp.active_desc);

  for (size_t i = 0; i < vp.active_desc.size(); ++i) {
    auto& name = vp.active_desc[i].name;
    theta_true.push_back(vp.theta_true[i]);
    if (ires.success) {
      theta_rec.push_back(ires.params[i]);
      err.push_back(abs_err[i]);
      err_phys.push_back(abs_err_phys[i]);
      rec_phys.push_back(get_param_value(recovered, name));
      true_phys.push_back(get_param_value(vp.true_params, name));
    } else {
      theta_rec.push_back(nullptr);
      err.push_back(nullptr);
      err_phys.push_back(nullptr);
      rec_phys.push_back(nullptr);
      true_phys.push_back(get_param_value(vp.true_params, name));
    }
  }
  j["theta_true"] = std::move(theta_true);
  j["theta_rec"] = std::move(theta_rec);
  j["abs_err_log"] = std::move(err);
  j["abs_err_phys"] = std::move(err_phys);
  j["param_true_phys"] = std::move(true_phys);
  j["param_rec_phys"] = std::move(rec_phys);

  return j;
}

// ===========================================================================
// Noise presets
// ===========================================================================
static const std::unordered_map<std::string, std::string> kPresetHelp = {
  {"cgm", "CGM σ=3, no dropouts, no meal errors"},
  {"dropout", "CGM σ=3, max_dropout=15min, p=0.005"},
  {"hetero", "Heteroskedastic: fixed σ=2, rel=5%"},
  {"meals", "CGM σ=3, meal skip=20%, shift=15min, amount=15%"},
  {"all", "Combined: hetero + dropout + meal errors"},
  {"none", "No noise at all (clean data)"},
};

static void print_presets() {
  std::cerr << "  Presets:\n";
  for (auto& [name, desc] : kPresetHelp)
    std::cerr << "    " << name << " — " << desc << "\n";
}

static void apply_noise_preset(VirtualPatientConfig& cfg, const std::string& name) {
  auto reset = [](VirtualPatientConfig& c) {
    c.cgm_noise = CGMNoiseConfig{};
    c.meal_errors = MealErrorConfig{};
  };

  if (name == "cgm") {
    reset(cfg);
    cfg.cgm_noise.fixed_sd = 3.0;
  } else if (name == "dropout") {
    reset(cfg);
    cfg.cgm_noise.fixed_sd = 3.0;
    cfg.cgm_noise.max_dropout_min = 15;
    cfg.cgm_noise.dropout_prob = 0.005;
  } else if (name == "hetero") {
    reset(cfg);
    cfg.cgm_noise.fixed_sd = 2.0;
    cfg.cgm_noise.relative_sd = 5.0;
  } else if (name == "meals") {
    reset(cfg);
    cfg.cgm_noise.fixed_sd = 3.0;
    cfg.meal_errors.skip_prob = 0.20;
    cfg.meal_errors.timing_shift_sd = 15.0;
    cfg.meal_errors.amount_noise_sd = 0.15;
  } else if (name == "all") {
    reset(cfg);
    cfg.cgm_noise.fixed_sd = 2.0;
    cfg.cgm_noise.relative_sd = 3.0;
    cfg.cgm_noise.max_dropout_min = 10;
    cfg.cgm_noise.dropout_prob = 0.002;
    cfg.meal_errors.skip_prob = 0.10;
    cfg.meal_errors.timing_shift_sd = 10.0;
    cfg.meal_errors.amount_noise_sd = 0.10;
  } else if (name == "none") {
    reset(cfg);
    cfg.cgm_noise.fixed_sd = 0.0;
  } else if (name != "custom") {
    std::cerr << "WARNING: unknown noise preset \"" << name << "\", using custom\n";
  }
}

// ===========================================================================
// Load JSON config file — overlays onto existing VirtualPatientConfig
// ===========================================================================
static void load_json_config(VirtualPatientConfig& cfg, const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "ERROR: cannot open config file " << path << "\n";
    std::exit(1);
  }

  json j;
  try {
    file >> j;
  } catch (const json::parse_error& e) {
    std::cerr << "ERROR: JSON parse error in " << path << ": " << e.what() << "\n";
    std::exit(1);
  }

  // CGM noise
  if (auto it = j.find("cgm_noise"); it != j.end()) {
    if (auto v = it->find("fixed_sd"); v != it->end()) cfg.cgm_noise.fixed_sd = v->get<double>();
    if (auto v = it->find("relative_sd"); v != it->end()) cfg.cgm_noise.relative_sd = v->get<double>();
    if (auto v = it->find("max_dropout_min"); v != it->end()) cfg.cgm_noise.max_dropout_min = v->get<int>();
    if (auto v = it->find("dropout_prob"); v != it->end()) cfg.cgm_noise.dropout_prob = v->get<double>();
  }

  // Meal errors
  if (auto it = j.find("meal_errors"); it != j.end()) {
    if (auto v = it->find("skip_prob"); v != it->end()) cfg.meal_errors.skip_prob = v->get<double>();
    if (auto v = it->find("timing_shift_sd"); v != it->end()) cfg.meal_errors.timing_shift_sd = v->get<double>();
    if (auto v = it->find("amount_noise_sd"); v != it->end()) cfg.meal_errors.amount_noise_sd = v->get<double>();
  }

  // Optimizer
  if (auto it = j.find("optimizer"); it != j.end()) {
    if (auto v = it->find("n_cmaes_restarts"); v != it->end()) cfg.n_cmaes_restarts = v->get<int>();
    if (auto v = it->find("cmaes_evals"); v != it->end()) cfg.cmaes_evals = v->get<int>();
  }

  // Simulation
  if (auto it = j.find("simulation"); it != j.end()) {
    if (auto v = it->find("duration_min"); v != it->end()) cfg.duration_min = v->get<int>();
    if (auto v = it->find("cgm_interval"); v != it->end()) cfg.cgm_interval = v->get<int>();
    if (auto v = it->find("perturbation_std"); v != it->end()) cfg.perturbation_std = v->get<double>();
  }

  // Daily meals
  if (auto it = j.find("daily_meals"); it != j.end() && it->is_array() && !it->empty()) {
    cfg.daily_meals.clear();
    for (const auto& m : *it) {
      VirtualPatientConfig::DailyMeal meal;
      meal.time_min = m["time_min"].get<int>();
      // support both "cho_g" and "cho_mg" for convenience
      if (m.contains("cho_g")) meal.cho_mg = m["cho_g"].get<double>() * 1000.0;
      else meal.cho_mg = m["cho_mg"].get<double>();
      // support both "bolus_u" and "bolus_dose_pmol_kg"
      if (m.contains("bolus_u")) meal.bolus_dose_pmol_kg = m["bolus_u"].get<double>() * 6000.0 / 75.0;
      else meal.bolus_dose_pmol_kg = m["bolus_dose_pmol_kg"].get<double>();
      cfg.daily_meals.push_back(meal);
    }
  }
}

// ===========================================================================
// CLI overrides — applied after JSON config and noise preset
// ===========================================================================
struct CLIConfig {
  int n_patients = 20;
  int base_seed = 42;
  bool verbose = false;
  std::string csv_path = "crossval_results.csv";
  std::string json_path = "crossval_results.json";
  std::string config_path;
  std::string noise_preset;
  std::string params_config_path;

  // CGM overrides (NaN = not specified)
  double cgm_fixed_sd = std::numeric_limits<double>::quiet_NaN();
  double cgm_rel_sd = std::numeric_limits<double>::quiet_NaN();
  int cgm_dropout_min = -1;
  double cgm_dropout_prob = std::numeric_limits<double>::quiet_NaN();

  // Meal overrides
  double meal_skip_prob = std::numeric_limits<double>::quiet_NaN();
  double meal_timing_sd = std::numeric_limits<double>::quiet_NaN();
  double meal_amount_sd = std::numeric_limits<double>::quiet_NaN();

  // Simulation overrides
  int duration_min = -1;
  int cgm_interval = -1;
  double perturbation_std = std::numeric_limits<double>::quiet_NaN();

  // Optimizer overrides
  int restarts = -1;
  int evals = -1;
  int lm_iters = -1;
  double cmaes_lower = std::numeric_limits<double>::quiet_NaN();
  double cmaes_upper = std::numeric_limits<double>::quiet_NaN();
  double prior_std = std::numeric_limits<double>::quiet_NaN();
};

static void apply_cli_overrides(VirtualPatientConfig& cfg, const CLIConfig& cli) {
  if (!std::isnan(cli.cgm_fixed_sd)) cfg.cgm_noise.fixed_sd = cli.cgm_fixed_sd;
  if (!std::isnan(cli.cgm_rel_sd)) cfg.cgm_noise.relative_sd = cli.cgm_rel_sd;
  if (cli.cgm_dropout_min >= 0) cfg.cgm_noise.max_dropout_min = cli.cgm_dropout_min;
  if (!std::isnan(cli.cgm_dropout_prob)) cfg.cgm_noise.dropout_prob = cli.cgm_dropout_prob;
  if (!std::isnan(cli.meal_skip_prob)) cfg.meal_errors.skip_prob = cli.meal_skip_prob;
  if (!std::isnan(cli.meal_timing_sd)) cfg.meal_errors.timing_shift_sd = cli.meal_timing_sd;
  if (!std::isnan(cli.meal_amount_sd)) cfg.meal_errors.amount_noise_sd = cli.meal_amount_sd;
  if (cli.duration_min > 0) cfg.duration_min = cli.duration_min;
  if (cli.cgm_interval > 0) cfg.cgm_interval = cli.cgm_interval;
  if (!std::isnan(cli.perturbation_std)) cfg.perturbation_std = cli.perturbation_std;
  if (cli.restarts > 0) cfg.n_cmaes_restarts = cli.restarts;
  if (cli.evals > 0) cfg.cmaes_evals = cli.evals;
}

static bool is_number(const std::string& s) {
  char* end = nullptr;
  std::strtod(s.c_str(), &end);
  return end != s.c_str() && *end == '\0';
}

// ===========================================================================
// Help
// ===========================================================================
static void print_help(const char* prog) {
  std::cerr << "Usage: " << prog << " [options]\n\n"
            << "General:\n"
            << "  --patients N        Number of virtual patients (default: 20)\n"
            << "  --seed N            Base random seed (default: 42)\n"
            << "  --verbose           Show CMA-ES/Ceres output during identification\n"
            << "  --csv FILE          Output CSV path (default: crossval_results.csv)\n"
            << "  --json FILE         Output JSON path (default: crossval_results.json)\n"
            << "  --help              Show this message\n\n"
            << "Configuration file:\n"
            << "  --config FILE       Load JSON config file (overlaid before presets & flags)\n"
            << "  --params-config FILE  JSON parameter descriptors (default: config/default_params.json)\n\n"
            << "Noise preset (overrides CGM + meal sections):\n"
            << "  --noise-preset NAME  " << kPresetHelp.at("cgm") << "\n";
  print_presets();
  std::cerr << "\nCGM noise overrides:\n"
            << "  --cgm-fixed-sd F    Fixed std dev [mg/dL] (default: varies by preset)\n"
            << "  --cgm-rel-sd F      Relative std dev [%% of Gsc] (default: varies)\n"
            << "  --cgm-dropout-min N Max dropout duration [min] (default: varies)\n"
            << "  --cgm-dropout-prob F Probability of starting a dropout (default: varies)\n\n"
            << "Meal error overrides:\n"
            << "  --meal-skip-prob F     Probability of skipping a meal (default: varies)\n"
            << "  --meal-timing-shift F  Std dev of time reporting error [min] (default: varies)\n"
            << "  --meal-amount-noise F  Std dev of CHO fraction (default: varies)\n\n"
            << "Simulation overrides:\n"
            << "  --duration-min N     Simulation length [min] (default: 4320)\n"
            << "  --cgm-interval N     CGM sampling interval [min] (default: 5)\n"
            << "  --perturbation-std F Std dev of true θ in log-space (default: 0.3)\n\n"
            << "Optimizer overrides:\n"
            << "  --restarts N         CMA-ES restarts per patient (default: 3)\n"
            << "  --evals N            CMA-ES evaluations per restart (default: 1000)\n"
            << "  --lm-iters N         Max Levenberg-Marquardt iterations (default: 100)\n"
            << "  --cmaes-lower F      CMA-ES lower bound in log-space (default: -3)\n"
            << "  --cmaes-upper F      CMA-ES upper bound in log-space (default: 3)\n"
            << "  --prior-std F        Prior std for all active params (default: from config)\n";
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char** argv) {
  std::cout << std::fixed << std::setprecision(4);

  if (argc == 1) {
    std::cerr << "Use --help for usage information.\n";
  }

  // ---- Parse CLI into CLIConfig ----
  CLIConfig cli;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) { std::cerr << "ERROR: " << arg << " requires a value\n"; std::exit(1); }
      return argv[i];
    };

    if (arg == "--patients") cli.n_patients = std::stoi(next());
    else if (arg == "--seed") cli.base_seed = std::stoi(next());
    else if (arg == "--verbose") cli.verbose = true;
    else if (arg == "--csv") cli.csv_path = next();
    else if (arg == "--json") cli.json_path = next();
    else if (arg == "--config") cli.config_path = next();
    else if (arg == "--params-config") cli.params_config_path = next();
    else if (arg == "--noise-preset") cli.noise_preset = next();
    else if (arg == "--cgm-fixed-sd") cli.cgm_fixed_sd = std::stod(next());
    else if (arg == "--cgm-rel-sd") cli.cgm_rel_sd = std::stod(next());
    else if (arg == "--cgm-dropout-min") cli.cgm_dropout_min = std::stoi(next());
    else if (arg == "--cgm-dropout-prob") cli.cgm_dropout_prob = std::stod(next());
    else if (arg == "--meal-skip-prob") cli.meal_skip_prob = std::stod(next());
    else if (arg == "--meal-timing-shift") cli.meal_timing_sd = std::stod(next());
    else if (arg == "--meal-amount-noise") cli.meal_amount_sd = std::stod(next());
    else if (arg == "--duration-min") cli.duration_min = std::stoi(next());
    else if (arg == "--cgm-interval") cli.cgm_interval = std::stoi(next());
    else if (arg == "--perturbation-std") cli.perturbation_std = std::stod(next());
    else if (arg == "--restarts") cli.restarts = std::stoi(next());
    else if (arg == "--evals") cli.evals = std::stoi(next());
    else if (arg == "--lm-iters") cli.lm_iters = std::stoi(next());
    else if (arg == "--cmaes-lower") cli.cmaes_lower = std::stod(next());
    else if (arg == "--cmaes-upper") cli.cmaes_upper = std::stod(next());
    else if (arg == "--prior-std") cli.prior_std = std::stod(next());
    else if (arg == "--help") { print_help(argv[0]); return 0; }
    else if (arg.find("--") == 0) {
      std::cerr << "WARNING: unknown flag " << arg << ", ignoring\n";
    }
  }

  // ---- Build VirtualPatientConfig ----
  // Order: defaults → JSON config → noise preset → CLI overrides
  auto cfg = default_config();
  cfg.seed = cli.base_seed;

  if (!cli.config_path.empty())
    load_json_config(cfg, cli.config_path);

  if (!cli.noise_preset.empty())
    apply_noise_preset(cfg, cli.noise_preset);

  apply_cli_overrides(cfg, cli);
  // cli fields that directly map to cfg
  cfg.seed = cli.base_seed;
  cfg.n_cmaes_restarts = cfg.n_cmaes_restarts;  // already set
  cfg.cmaes_evals = cfg.cmaes_evals;

  // Custom param descriptors path
  cfg.params_config_path = cli.params_config_path;

  // Unpack optimizer params for the pipeline call
  int lm_max_iters = (cli.lm_iters > 0) ? cli.lm_iters : 100;
  double cmaes_lower = std::isnan(cli.cmaes_lower) ? -3.0 : cli.cmaes_lower;
  double cmaes_upper = std::isnan(cli.cmaes_upper) ? 3.0 : cli.cmaes_upper;
  double prior_std = std::isnan(cli.prior_std) ? 1.0 : cli.prior_std;

  // ---- Print config ----
  auto preset_it = kPresetHelp.find(cli.noise_preset);
  std::string preset_desc = (preset_it != kPresetHelp.end()) ? preset_it->second : "custom";
  std::string preset_name = cli.noise_preset.empty() ? "(none)" : cli.noise_preset;
  std::cout << "=== UVA-Padova Cross-Validation ===\n"
            << "  Patients:       " << cli.n_patients << "\n"
            << "  Seed:           " << cli.base_seed << "\n"
            << "  Preset:         " << preset_name << " (" << preset_desc << ")\n"
            << "  Config file:    " << (cli.config_path.empty() ? "(none)" : cli.config_path) << "\n"
            << "  Params config:  " << (cli.params_config_path.empty() ? "(default)" : cli.params_config_path) << "\n"
            << "  Duration:       " << cfg.duration_min << " min (" << (cfg.duration_min / 1440) << " days)\n"
            << "  CGM interval:   " << cfg.cgm_interval << " min\n"
            << "  ╶ CGM fixed σ:  " << cfg.cgm_noise.fixed_sd << " mg/dL\n"
            << "  ╶ CGM rel σ:    " << cfg.cgm_noise.relative_sd << " %\n"
            << "  ╶ CGM dropout:  " << (cfg.cgm_noise.max_dropout_min > 0 ?
                std::to_string(cfg.cgm_noise.max_dropout_min) + " min, p=" + std::to_string(cfg.cgm_noise.dropout_prob) : "none") << "\n"
            << "  ╶ Meal skip:    " << (cfg.meal_errors.skip_prob > 0 ?
                std::to_string(cfg.meal_errors.skip_prob * 100) + "%" : "none") << "\n"
            << "  ╶ Meal timing:  " << (cfg.meal_errors.timing_shift_sd > 0 ?
                std::to_string(cfg.meal_errors.timing_shift_sd) + " min" : "none") << "\n"
            << "  ╶ Meal amount:  " << (cfg.meal_errors.amount_noise_sd > 0 ?
                std::to_string(cfg.meal_errors.amount_noise_sd * 100) + "%" : "none") << "\n"
            << "  Restarts:       " << cfg.n_cmaes_restarts << "\n"
            << "  CMA-ES evals:   " << cfg.cmaes_evals << "\n"
            << "  LM max iters:   " << lm_max_iters << "\n"
            << "  CMAES bounds:   [" << cmaes_lower << ", " << cmaes_upper << "]\n"
            << "  Prior std:      " << prior_std << "\n"
            << "  Verbose:        " << (cli.verbose ? "yes" : "no (output suppressed)") << "\n"
            << "  CSV:            " << cli.csv_path << "\n"
            << "  JSON:           " << cli.json_path << "\n\n";

  // ---- Open output files ----
  std::ofstream csv_file(cli.csv_path);
  if (!csv_file.is_open()) {
    std::cerr << "ERROR: cannot open " << cli.csv_path << " for writing\n";
    return 1;
  }

  // ---- Aggregated state ----
  int n_success = 0;
  int n_gateA_passed = 0;
  int n_gateB_passed = 0;
  std::vector<double> all_abs_err;
  json results_json = json::array();
  bool header_written = false;
  std::vector<ParamDescriptor> global_active_desc;

  // ---- Patient loop ----
  for (int pid = 0; pid < cli.n_patients; ++pid) {
    auto t_start = std::chrono::steady_clock::now();

    std::cout << "[" << (pid + 1) << "/" << cli.n_patients << "] "
              << "Generating patient " << pid << "...\n";

    // 1. Generate virtual patient
    auto vp = generate_virtual_patient(cfg, pid);
    if (global_active_desc.empty())
      global_active_desc = vp.active_desc;

    // Override prior_std if specified
    if (!std::isnan(cli.prior_std)) {
      for (auto& d : vp.active_desc)
        d.prior_std = prior_std;
    }

    // 2. Filter dropout zeros from data
    auto id_data = filter_cgm_zeros(vp.data);

    // 3. Run identification
    std::cout << "  Identifying (" << id_data.cgm_time.size()
              << " obs, " << id_data.meal_time.size() << " meals)...\n";

    IdentificationPipeline<>::Result ires;
    if (cli.verbose) {
      ires = IdentificationPipeline<>::run(
          id_data, vp.base_params, vp.active_desc,
          cfg.n_cmaes_restarts, cfg.cmaes_evals,
          lm_max_iters, cmaes_lower, cmaes_upper);
    } else {
      CoutSilencer silence;
      ires = IdentificationPipeline<>::run(
          id_data, vp.base_params, vp.active_desc,
          cfg.n_cmaes_restarts, cfg.cmaes_evals,
          lm_max_iters, cmaes_lower, cmaes_upper);
    }

    std::cout << "  [debug] pipeline returned\n";
    std::cout << "  Result: success=" << (ires.success ? "YES" : "NO")
              << " cost=" << ires.cost
              << " evals=" << ires.total_evals << "\n";

    // 4. Gate A
    GateAResult gateA;
    if (ires.success)
      gateA = diagnose_multiplicity(ires.all_restarts, vp.active_desc);
    if (gateA.passed) n_gateA_passed++;
    std::cout << "  [debug] Gate A done\n";

    // 5. Gate B
    GateBResult gateB;
    if (ires.success) {
      ObservationCost<> obs_cost(id_data, vp.base_params, vp.active_desc);
      std::cout << "  [debug] Gate B starting...\n";
      gateB = diagnose_structural_adequacy(
          id_data.cgm_time, id_data.cgm_value, id_data.meal_time,
          ires.params.data(), obs_cost);
      std::cout << "  [debug] Gate B done\n";
    }
    if (gateB.passed) n_gateB_passed++;

    // 6. Gate C
    GateCResult gateC;
    if (ires.success) {
      ObservationCost<> obs_cost_gc(id_data, vp.base_params, vp.active_desc);
      std::cout << "  [debug] Gate C starting...\n";
      gateC = diagnose_observability(ires.params.data(), obs_cost_gc, vp.active_desc);
      std::cout << "  [debug] Gate C done\n";
    }

    // 7. Recovery metrics
    std::cout << "  [debug] computing recovery metrics...\n";
    std::vector<double> abs_err(vp.active_desc.size(), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> abs_err_phys(vp.active_desc.size(), std::numeric_limits<double>::quiet_NaN());
    Params recovered_params;

    if (ires.success) {
      recovered_params = ires.to_params(vp.base_params, vp.active_desc);
      for (size_t i = 0; i < vp.active_desc.size(); ++i) {
        double t = vp.theta_true[i];
        double r = ires.params[i];
        abs_err[i] = std::abs(r - t);
        double pt = get_param_value(vp.true_params, vp.active_desc[i].name);
        double pr = get_param_value(recovered_params, vp.active_desc[i].name);
        abs_err_phys[i] = std::abs(pr - pt);
        all_abs_err.push_back(abs_err[i]);
      }
      n_success++;
    }
    std::cout << "  [debug] recovery metrics done\n";

    // Print gate summary
    std::cout << "  Gates: A=" << (gateA.passed ? "PASS" : "FAIL")
              << "(" << gateA.n_clusters << " clusters)"
              << " B=" << (gateB.passed ? "PASS" : "FAIL")
              << "(" << (gateB.ljung_box.significant ? "autocorr " : "")
              << (gateB.regime_bias_detected ? "bias" : "") << ")"
              << " C=" << std::count(gateC.prior_dominated.begin(), gateC.prior_dominated.end(), true)
              << " dominated\n";

    // 8. Write CSV
    if (!header_written) {
      write_csv_header(csv_file, vp.active_desc);
      header_written = true;
    }
    write_csv_row(csv_file, pid, ires, vp, gateA, gateB, gateC, abs_err, abs_err_phys);
    csv_file.flush();

    // 9. Build JSON
    auto t_end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
    results_json.push_back(build_json_result(pid, ires, vp, gateA, gateB, gateC,
                                             abs_err, abs_err_phys, elapsed_s));

    std::cout << "  Elapsed: " << elapsed_s << " s\n\n";
  }

  csv_file.close();

  // ---- Summary ----
  double mean_abs_err = 0.0;
  if (!all_abs_err.empty()) {
    double sum = 0.0;
    for (double v : all_abs_err) sum += v;
    mean_abs_err = sum / all_abs_err.size();
  }

  std::cout << "=== Summary ===\n"
            << "  Successful identifications: " << n_success << "/" << cli.n_patients << "\n"
            << "  Gate A pass rate:  " << (100.0 * n_gateA_passed / cli.n_patients) << "%\n"
            << "  Gate B pass rate:  " << (100.0 * n_gateB_passed / cli.n_patients) << "%\n"
            << "  Mean |θ_err|:      " << mean_abs_err << "\n"
            << "  Results written to: " << cli.csv_path << ", " << cli.json_path << "\n";

  // ---- Write JSON ----
  json jout;
  jout["config"]["n_patients"] = cli.n_patients;
  jout["config"]["noise_preset"] = cli.noise_preset;
  jout["config"]["cgm_noise"]["fixed_sd"] = cfg.cgm_noise.fixed_sd;
  jout["config"]["cgm_noise"]["relative_sd"] = cfg.cgm_noise.relative_sd;
  jout["config"]["cgm_noise"]["max_dropout_min"] = cfg.cgm_noise.max_dropout_min;
  jout["config"]["cgm_noise"]["dropout_prob"] = cfg.cgm_noise.dropout_prob;
  jout["config"]["meal_errors"]["skip_prob"] = cfg.meal_errors.skip_prob;
  jout["config"]["meal_errors"]["timing_shift_sd"] = cfg.meal_errors.timing_shift_sd;
  jout["config"]["meal_errors"]["amount_noise_sd"] = cfg.meal_errors.amount_noise_sd;
  jout["config"]["optimizer"]["restarts"] = cfg.n_cmaes_restarts;
  jout["config"]["optimizer"]["cmaes_evals"] = cfg.cmaes_evals;
  jout["config"]["optimizer"]["lm_max_iters"] = lm_max_iters;
  jout["config"]["optimizer"]["cmaes_lower"] = cmaes_lower;
  jout["config"]["optimizer"]["cmaes_upper"] = cmaes_upper;
  jout["config"]["optimizer"]["prior_std"] = prior_std;

  jout["summary"]["n_success"] = n_success;
  jout["summary"]["n_gateA_passed"] = n_gateA_passed;
  jout["summary"]["n_gateB_passed"] = n_gateB_passed;
  jout["summary"]["mean_abs_err_log"] = mean_abs_err;

  if (!global_active_desc.empty()) {
    json pnames = json::array();
    for (auto& d : global_active_desc)
      pnames.push_back(d.name);
    jout["param_names"] = std::move(pnames);
  }

  jout["results"] = std::move(results_json);

  std::ofstream jfile(cli.json_path);
  jfile << jout.dump(2);
  jfile.close();

  return 0;
}
