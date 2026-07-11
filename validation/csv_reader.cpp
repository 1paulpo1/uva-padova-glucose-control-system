#include "csv_reader.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "uva_padova/integrals.hpp"
#include "uva_padova/constants.hpp"

namespace uva_padova {

IdentificationData RealDataReader::load_from_csv(
    const std::string& cgm_csv_path,
    const std::string& basal_csv_path,
    const Params& base_params)
{
  IdentificationData data;

  // ---- Read CGM / meals / boluses ----
  std::ifstream cgm_file(cgm_csv_path);
  if (!cgm_file.is_open()) {
    std::cerr << "Error: could not open " << cgm_csv_path << "\n";
    return data;
  }

  std::string line;
  // header
  std::getline(cgm_file, line);

  while (std::getline(cgm_file, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string cell;

    std::getline(ss, cell, ',');
    double t = std::stod(cell);

    std::getline(ss, cell, ',');
    double cgm_val = std::stod(cell);

    std::getline(ss, cell, ',');
    double meal_mg = std::stod(cell);

    std::getline(ss, cell, ',');
    double bolus_pmol_kg = std::stod(cell);

    if (cgm_val > 0.0) {
      data.cgm_time.push_back(t);
      data.cgm_value.push_back(cgm_val);
    }
    if (meal_mg > 0.0) {
      data.meal_time.push_back(t);
      data.meal_cho.push_back(meal_mg);
    }
    if (bolus_pmol_kg > 0.0) {
      data.bolus_time.push_back(t);
      data.bolus_dose.push_back(bolus_pmol_kg);
    }
  }

  // ---- Read basal profile ----
  std::ifstream basal_file(basal_csv_path);
  if (!basal_file.is_open()) {
    std::cerr << "Error: could not open " << basal_csv_path << "\n";
    return data;
  }

  std::vector< std::pair<double, double> > basal_segments; // (t_min, rate_u_per_hr)

  std::getline(basal_file, line); // header
  while (std::getline(basal_file, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string cell;

    std::getline(ss, cell, ',');
    double t = std::stod(cell);

    std::getline(ss, cell, ',');
    double rate_u_per_hr = std::stod(cell);

    basal_segments.emplace_back(t, rate_u_per_hr);
  }

  if (basal_segments.empty()) {
    std::cerr << "Warning: no basal segments found\n";
    return data;
  }

  // ---- Step-change basal encoding ----
  // Compute steady-state insulin rate for the default params
  State ss_state;
  double insulin_ss = SteadyStateSolver::find(ss_state, base_params, base_params.Gb);
  double R_ss_pmol_kg_min = insulin_ss; // already pmol/kg/min from find()

  // Basal segments to step changes
  //   insulin_time[i] = t_i
  //   insulin_rate[i] = conv(rate[i]) - conv(rate[i-1])
  //   with conv(rate[-1]) = R_ss
  double prev_rate_pmol_kg_min = R_ss_pmol_kg_min;

  for (const auto& seg : basal_segments) {
    double rate_pmol_kg_min = constants::u_per_hr_to_pmol_per_kg_per_min(
        seg.second, base_params.BW);
    double delta = rate_pmol_kg_min - prev_rate_pmol_kg_min;

    data.insulin_time.push_back(seg.first);
    data.insulin_rate.push_back(delta);

    prev_rate_pmol_kg_min = rate_pmol_kg_min;
  }

  // ---- total_min ----
  if (!data.cgm_time.empty()) {
    data.total_min_ = static_cast<int>(data.cgm_time.back());
  }

  // ---- Summary ----
  std::cout << "[RealDataReader] loaded "
            << data.cgm_time.size() << " CGM, "
            << data.meal_time.size() << " meals, "
            << data.bolus_time.size() << " boluses, "
            << basal_segments.size() << " basal segments"
            << " (" << data.total_min_ << " min)\n";

  return data;
}

} // namespace uva_padova
