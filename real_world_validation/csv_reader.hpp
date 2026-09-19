#pragma once

#include <string>
#include "uva_padova/cost_function.hpp"
#include "uva_padova/types.hpp"

namespace uva_padova {

struct RealDataReader {
  // Load a single day of patient data from CSV files produced by
  // parse_patient_data.py.
  //
  // cgm_csv_path:  columns t_min, cgm_mgdl, meal_mg, bolus_pmol_kg
  // basal_csv_path: columns t_min, basal_u_per_hr
  // base_params:   needed for steady-state rate computation (step-change encoding)
  // desc:          param descriptors (to determine n_params, names)
  static IdentificationData load_from_csv(
      const std::string& cgm_csv_path,
      const std::string& basal_csv_path,
      const Params& base_params);
};

} // namespace uva_padova
