#pragma once

#include <string>
#include <vector>
#include "uva_padova/cost_function.hpp"

namespace uva_padova {

// Load full parameter descriptors from a JSON configuration file.
// Expected JSON format:
// {
//   "parameters": [
//     {"name": "kp1"},
//     {"name": "Km0", "fixed": true, "prior_mean": 0.0, "prior_std": 2.0, "initial": 0.0},
//     ...
//   ]
// }
// All fields except "name" are optional:
//   - fixed:       bool, default false
//   - prior_mean:  double (log-space), default 0.0 (i.e. log(1)=0 for identity multiplier)
//   - prior_std:   double (log-space), default 2.0
//   - initial:     double (log-space), default 0.0
//   - online:      bool, default false
std::vector<ParamDescriptor> load_param_descriptors(const std::string& json_path);

// Return only non-fixed descriptors (for building the optimisation theta vector).
std::vector<ParamDescriptor> active_descriptors(const std::vector<ParamDescriptor>& all);

// Load the default configuration (default_config_5_active.json in the config directory).
// Returns an empty vector if the file cannot be found.
std::vector<ParamDescriptor> load_default_param_descriptors();

} // namespace uva_padova
