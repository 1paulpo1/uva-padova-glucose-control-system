#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "virtual_patient.hpp"
#include "uva_padova/config_loader.hpp"
#include "uva_padova/optimizer.hpp"
#include "uva_padova/diagnostics.hpp"
#include "uva_padova/cost_function.hpp"
#include "uva_padova/constants.hpp"

using namespace uva_padova;
using namespace uva_padova::crossval;
using json = nlohmann::json;

struct CoutSilencer {
  std::streambuf* old;
  std::ofstream null;
  CoutSilencer() : old(std::cout.rdbuf()), null("/dev/null") {
    if (null.is_open()) std::cout.rdbuf(null.rdbuf());
  }
  ~CoutSilencer() { std::cout.rdbuf(old); }
};

struct TopologyConfig {
  int n_patients{1};
  int base_seed{42};
  bool verbose{false};
  std::string json_path{"topology_results.json"};
  std::string params_config_path;

  int n_starts{200};
  std::string start_dist{"normal"};
  double start_sigma{1.0};
  double start_lower{-3.0};
  double start_upper{3.0};

  double dbscan_eps{2.0};
  int dbscan_minPts{2};
  double cluster_thresh{2.0};

  int lm_max_iters{100};

  VirtualPatientConfig vp_cfg;
};

static void print_help(const char* prog) {
  std::cerr << "Usage: " << prog << " [options]\n\n"
            << "General:\n"
            << "  --patients N        Number of virtual patients (default: 1)\n"
            << "  --seed N            Base random seed (default: 42)\n"
            << "  --verbose           Show Ceres output during LM\n"
            << "  --json FILE         Output JSON path (default: topology_results.json)\n"
            << "  --params-config FILE  JSON param descriptors (default: config/default_params.json)\n"
            << "  --help              Show this message\n\n"
            << "Multi-start:\n"
            << "  --starts N          Number of LM runs per patient (default: 200)\n"
            << "  --start-dist TYPE   Init distribution: normal or uniform (default: normal)\n"
            << "  --start-sigma F     Std dev for normal init (default: 1.0)\n"
            << "  --start-lower F     Lower bound for uniform init (default: -3.0)\n"
            << "  --start-upper F     Upper bound for uniform init (default: 3.0)\n\n"
            << "Clustering:\n"
            << "  --dbscan-eps F      DBSCAN neighbourhood radius (default: 2.0)\n"
            << "  --dbscan-minpts N   Min points for core point (default: 2)\n"
            << "  --cluster-thresh F  Max distance to theta_true for near-truth label (default: 2.0)\n\n"
            << "Simulation:\n"
            << "  --duration-min N    Simulation length [min] (default: 4320)\n"
            << "  --perturbation-std F  Std dev of true theta in log-space (default: 0.3)\n"
            << "  --cgm-interval N    CGM sampling interval [min] (default: 5)\n\n"
            << "Optimizer:\n"
            << "  --lm-iters N        Max LM iterations (default: 100)\n";
}

int main(int argc, char** argv) {
  std::cout << std::fixed << std::setprecision(4);

  TopologyConfig cfg;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) { std::cerr << "ERROR: " << arg << " requires a value\n"; return ""; }
      return argv[i];
    };

    if (arg == "--patients") cfg.n_patients = std::stoi(next());
    else if (arg == "--seed") cfg.base_seed = std::stoi(next());
    else if (arg == "--verbose") cfg.verbose = true;
    else if (arg == "--json") cfg.json_path = next();
    else if (arg == "--params-config") cfg.params_config_path = next();
    else if (arg == "--starts") cfg.n_starts = std::stoi(next());
    else if (arg == "--start-dist") cfg.start_dist = next();
    else if (arg == "--start-sigma") cfg.start_sigma = std::stod(next());
    else if (arg == "--start-lower") cfg.start_lower = std::stod(next());
    else if (arg == "--start-upper") cfg.start_upper = std::stod(next());
    else if (arg == "--dbscan-eps") cfg.dbscan_eps = std::stod(next());
    else if (arg == "--dbscan-minpts") cfg.dbscan_minPts = std::stoi(next());
    else if (arg == "--cluster-thresh") cfg.cluster_thresh = std::stod(next());
    else if (arg == "--duration-min") cfg.vp_cfg.duration_min = std::stoi(next());
    else if (arg == "--perturbation-std") cfg.vp_cfg.perturbation_std = std::stod(next());
    else if (arg == "--cgm-interval") cfg.vp_cfg.cgm_interval = std::stoi(next());
    else if (arg == "--lm-iters") cfg.lm_max_iters = std::stoi(next());
    else if (arg == "--help") { print_help(argv[0]); return 0; }
    else if (arg.find("--") == 0) {
      std::cerr << "WARNING: unknown flag " << arg << ", ignoring\n";
    }
  }

  cfg.vp_cfg.seed = cfg.base_seed;
  cfg.vp_cfg.cgm_noise.fixed_sd = 3.0;
  cfg.vp_cfg.cgm_noise.relative_sd = 0.0;
  cfg.vp_cfg.cgm_noise.max_dropout_min = 0;
  cfg.vp_cfg.cgm_noise.dropout_prob = 0.0;
  cfg.vp_cfg.meal_errors = {};
  cfg.vp_cfg.params_config_path = cfg.params_config_path;

  std::cout << "=== Topology Explorer ===\n"
            << "  Patients:        " << cfg.n_patients << "\n"
            << "  Seed:            " << cfg.base_seed << "\n"
            << "  Multi-starts:    " << cfg.n_starts << " per patient\n"
            << "  Start dist:      " << cfg.start_dist
            << " (sigma=" << cfg.start_sigma
            << " bounds=[" << cfg.start_lower << "," << cfg.start_upper << "])\n"
            << "  DBSCAN eps:      " << cfg.dbscan_eps << "\n"
            << "  Cluster thresh:  " << cfg.cluster_thresh << "\n"
            << "  Duration:        " << cfg.vp_cfg.duration_min << " min ("
            << (cfg.vp_cfg.duration_min / 1440) << " days)\n"
            << "  CGM interval:    " << cfg.vp_cfg.cgm_interval << " min\n"
            << "  Perturbation:    " << cfg.vp_cfg.perturbation_std << "\n"
            << "  LM max iters:    " << cfg.lm_max_iters << "\n"
            << "  Params config:   "
            << (cfg.params_config_path.empty() ? "(default)" : cfg.params_config_path) << "\n"
            << "  JSON:            " << cfg.json_path << "\n\n";

  json jout;
  jout["config"]["n_patients"] = cfg.n_patients;
  jout["config"]["n_starts"] = cfg.n_starts;
  jout["config"]["start_distribution"] = cfg.start_dist;
  jout["config"]["start_sigma"] = cfg.start_sigma;
  jout["config"]["start_lower"] = cfg.start_lower;
  jout["config"]["start_upper"] = cfg.start_upper;
  jout["config"]["dbscan_eps"] = cfg.dbscan_eps;
  jout["config"]["dbscan_minPts"] = cfg.dbscan_minPts;
  jout["config"]["cluster_threshold"] = cfg.cluster_thresh;
  jout["config"]["duration_min"] = cfg.vp_cfg.duration_min;
  jout["config"]["perturbation_std"] = cfg.vp_cfg.perturbation_std;
  jout["config"]["lm_max_iters"] = cfg.lm_max_iters;

  json patients_arr = json::array();

  for (int pid = 0; pid < cfg.n_patients; ++pid) {
    auto t_start = std::chrono::steady_clock::now();

    std::cout << "[" << (pid + 1) << "/" << cfg.n_patients << "] "
              << "Generating patient " << pid << "...\n";

    auto vp = generate_virtual_patient(cfg.vp_cfg, pid);
    int dim = static_cast<int>(vp.active_desc.size());

    std::cout << "  Active params: " << dim << "\n";

    ObservationCost<> obs_cost(vp.data, vp.base_params, vp.active_desc);
    PriorCost prior_cost(vp.active_desc);

    double cost_at_true = obs_cost.compute_scalar_cost(vp.theta_true.data())
                        + prior_cost.compute_scalar_prior(vp.theta_true.data());
    std::vector<double> theta_zero(dim, 0.0);
    double cost_at_zero = obs_cost.compute_scalar_cost(theta_zero.data())
                        + prior_cost.compute_scalar_prior(theta_zero.data());

    std::cout << "  Cost at theta_true: " << cost_at_true << "\n"
              << "  Cost at theta=0:    " << cost_at_zero << "\n";

    struct StartRecord {
      std::vector<double> theta_init;
      std::vector<double> theta_final;
      double init_cost{1e20};
      double final_cost{1e20};
      int n_iters{0};
      bool converged{false};
    };

    std::vector<StartRecord> starts(cfg.n_starts);
    int n_converged = 0;

    for (int si = 0; si < cfg.n_starts; ++si) {
      int seed = cfg.base_seed + pid * 10000 + si;
      std::mt19937 rng(seed);

      std::vector<double> theta_init(dim);
      if (cfg.start_dist == "uniform") {
        std::uniform_real_distribution<double> dist(cfg.start_lower, cfg.start_upper);
        for (int i = 0; i < dim; ++i)
          theta_init[i] = dist(rng);
      } else {
        std::normal_distribution<double> dist(0.0, cfg.start_sigma);
        for (int i = 0; i < dim; ++i) {
          double v = dist(rng);
          v = std::max(cfg.start_lower, std::min(cfg.start_upper, v));
          theta_init[i] = v;
        }
      }

      double init_cost = obs_cost.compute_scalar_cost(theta_init.data())
                       + prior_cost.compute_scalar_prior(theta_init.data());

      starts[si].theta_init = theta_init;
      starts[si].init_cost = init_cost;

      auto lm_res = LMOptimizer::run(theta_init, obs_cost, prior_cost, cfg.lm_max_iters);

      starts[si].theta_final = lm_res.final_params;
      starts[si].final_cost = lm_res.final_cost;
      starts[si].n_iters = lm_res.n_iters;
      starts[si].converged = lm_res.converged;
      if (lm_res.converged) n_converged++;

      if ((si + 1) % 10 == 0) {
        std::cout << "  [" << (si + 1) << "/" << cfg.n_starts << "] "
                  << "converged=" << n_converged << " best=";
        double best_seen = 1e20;
        for (int j = 0; j <= si; ++j)
          if (starts[j].final_cost < best_seen) best_seen = starts[j].final_cost;
        std::cout << best_seen << "\n";
      }
    }

    std::vector<RestartRecord> final_records;
    final_records.reserve(cfg.n_starts);
    for (auto& s : starts)
      final_records.push_back({s.theta_final, s.final_cost});

    auto db = cluster_restarts(final_records, vp.active_desc,
                                cfg.dbscan_eps, cfg.dbscan_minPts);

    std::vector<int> cluster_sizes(db.n_clusters, 0);
    for (int i = 0; i < cfg.n_starts; ++i) {
      if (db.labels[i] >= 0)
        cluster_sizes[db.labels[i]]++;
    }
    int n_noise = 0;
    for (int i = 0; i < cfg.n_starts; ++i)
      if (db.labels[i] == -1) n_noise++;

    json patient_obj;
    patient_obj["patient_id"] = pid;
    patient_obj["n_active_params"] = dim;
    {
      json names = json::array();
      for (auto& d : vp.active_desc) names.push_back(d.name);
      patient_obj["param_names"] = names;
    }
    patient_obj["theta_true"] = json(vp.theta_true);
    patient_obj["cost_at_true"] = cost_at_true;
    patient_obj["cost_at_zero"] = cost_at_zero;
    patient_obj["n_starts"] = cfg.n_starts;
    patient_obj["n_clusters"] = db.n_clusters;
    patient_obj["n_noise_points"] = n_noise;

    json clusters_arr = json::array();
    for (int cid = 0; cid < db.n_clusters; ++cid) {
      std::vector<int> members;
      for (int i = 0; i < cfg.n_starts; ++i)
        if (db.labels[i] == cid) members.push_back(i);

      int nm = static_cast<int>(members.size());

      std::vector<double> centroid(dim, 0.0);
      for (int idx : members)
        for (int j = 0; j < dim; ++j)
          centroid[j] += starts[idx].theta_final[j];
      for (int j = 0; j < dim; ++j)
        centroid[j] /= nm;

      double dist_to_truth = 0.0;
      for (int j = 0; j < dim; ++j) {
        double d = (centroid[j] - vp.theta_true[j]) / vp.active_desc[j].prior_std;
        dist_to_truth += d * d;
      }
      dist_to_truth = std::sqrt(dist_to_truth);

      std::vector<double> costs;
      costs.reserve(nm);
      for (int idx : members)
        costs.push_back(starts[idx].final_cost);
      std::sort(costs.begin(), costs.end());

      double cost_min = costs.front();
      double cost_max = costs.back();
      double cost_sum = std::accumulate(costs.begin(), costs.end(), 0.0);
      double cost_mean = cost_sum / nm;
      double cost_median = costs[nm / 2];
      double cost_var = 0.0;
      for (auto c : costs) cost_var += (c - cost_mean) * (c - cost_mean);
      double cost_std = std::sqrt(cost_var / nm);

      double within_spread = 0.0;
      for (int idx : members) {
        double d2 = 0.0;
        for (int j = 0; j < dim; ++j) {
          double dd = (starts[idx].theta_final[j] - centroid[j]) / vp.active_desc[j].prior_std;
          d2 += dd * dd;
        }
        within_spread += std::sqrt(d2);
      }
      within_spread /= nm;

      json cluster_obj;
      cluster_obj["cluster_id"] = cid;
      cluster_obj["n_members"] = nm;
      cluster_obj["members"] = members;
      cluster_obj["centroid"] = centroid;
      cluster_obj["distance_to_truth"] = dist_to_truth;
      cluster_obj["near_truth"] = (dist_to_truth < cfg.cluster_thresh);
      cluster_obj["cost_min"] = cost_min;
      cluster_obj["cost_mean"] = cost_mean;
      cluster_obj["cost_median"] = cost_median;
      cluster_obj["cost_max"] = cost_max;
      cluster_obj["cost_std"] = cost_std;
      cluster_obj["within_cluster_spread"] = within_spread;
      clusters_arr.push_back(cluster_obj);

      std::cout << "  Cluster " << cid << ": n=" << nm
                << " dist_to_truth=" << dist_to_truth
                << (dist_to_truth < cfg.cluster_thresh ? " NEAR" : " FAR")
                << " cost=[" << cost_min << ", " << cost_max << "]"
                << " median=" << cost_median << "\n";
    }
    patient_obj["clusters"] = clusters_arr;

    json starts_arr = json::array();
    for (int i = 0; i < cfg.n_starts; ++i) {
      double d2 = 0.0;
      for (int j = 0; j < dim; ++j) {
        double dd = (starts[i].theta_final[j] - vp.theta_true[j]) / vp.active_desc[j].prior_std;
        d2 += dd * dd;
      }
      double dist_to_truth_i = std::sqrt(d2);

      json s_obj;
      s_obj["start_idx"] = i;
      s_obj["theta_init"] = starts[i].theta_init;
      s_obj["theta_final"] = starts[i].theta_final;
      s_obj["init_cost"] = starts[i].init_cost;
      s_obj["final_cost"] = starts[i].final_cost;
      s_obj["converged"] = starts[i].converged;
      s_obj["n_iters"] = starts[i].n_iters;
      s_obj["distance_to_truth"] = dist_to_truth_i;
      s_obj["cluster_id"] = db.labels[i];
      starts_arr.push_back(s_obj);
    }
    patient_obj["all_starts"] = starts_arr;

    patients_arr.push_back(patient_obj);

    double elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cout << "  Elapsed: " << elapsed_s << " s\n\n";
  }

  jout["results"] = patients_arr;

  {
    json summary_arr = json::array();
    for (int pid = 0; pid < cfg.n_patients; ++pid) {
      auto& p = patients_arr[pid];
      json s;
      s["patient_id"] = pid;
      s["n_clusters"] = p["n_clusters"];
      s["n_noise"] = p["n_noise_points"];
      s["cost_at_true"] = p["cost_at_true"];
      s["cost_at_zero"] = p["cost_at_zero"];

      double bc = 1e20;
      for (auto& c : p["clusters"])
        if (c["cost_min"].get<double>() < bc)
          bc = c["cost_min"].get<double>();
      s["best_cost_found"] = bc;

      int n_near = 0;
      for (auto& c : p["clusters"])
        if (c["near_truth"].get<bool>()) n_near += c["n_members"].get<int>();
      s["n_near_truth"] = n_near;

      summary_arr.push_back(s);
    }
    jout["summary"]["patients"] = summary_arr;
    jout["summary"]["total_patients"] = cfg.n_patients;
    jout["summary"]["total_starts"] = cfg.n_patients * cfg.n_starts;
  }

  std::ofstream json_file(cfg.json_path);
  if (json_file.is_open()) {
    json_file << jout.dump(2);
    json_file.close();
    std::cout << "Results written to: " << cfg.json_path << "\n";
  } else {
    std::cerr << "ERROR: cannot open " << cfg.json_path << " for writing\n";
    return 1;
  }

  return 0;
}
