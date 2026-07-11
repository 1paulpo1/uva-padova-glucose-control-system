#include "uva_padova/config_loader.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace uva_padova {

std::vector<ParamDescriptor> load_param_descriptors(const std::string& json_path) {
  std::ifstream file(json_path);
  if (!file.is_open()) {
    std::cerr << "config_loader: cannot open " << json_path << "\n";
    return {};
  }

  json j;
  try {
    file >> j;
  } catch (const json::parse_error& e) {
    std::cerr << "config_loader: JSON parse error in " << json_path
              << ": " << e.what() << "\n";
    return {};
  }

  auto params_it = j.find("parameters");
  if (params_it == j.end() || !params_it->is_array()) {
    std::cerr << "config_loader: missing or non-array 'parameters' key in "
              << json_path << "\n";
    return {};
  }

  std::vector<ParamDescriptor> descriptors;
  descriptors.reserve(params_it->size());

  for (const auto& item : *params_it) {
    ParamDescriptor desc;

    auto name_it = item.find("name");
    if (name_it == item.end() || !name_it->is_string()) {
      std::cerr << "config_loader: parameter entry missing 'name'\n";
      continue;
    }
    desc.name = name_it->get<std::string>();

    auto fixed_it = item.find("fixed");
    if (fixed_it != item.end() && fixed_it->is_boolean())
      desc.fixed = fixed_it->get<bool>();

    auto mean_it = item.find("prior_mean");
    if (mean_it != item.end() && mean_it->is_number())
      desc.prior_mean = mean_it->get<double>();

    auto std_it = item.find("prior_std");
    if (std_it != item.end() && std_it->is_number())
      desc.prior_std = std_it->get<double>();

    auto init_it = item.find("initial");
    if (init_it != item.end() && init_it->is_number())
      desc.initial_val = init_it->get<double>();

    descriptors.push_back(std::move(desc));
  }

  return descriptors;
}

std::vector<ParamDescriptor> active_descriptors(const std::vector<ParamDescriptor>& all) {
  std::vector<ParamDescriptor> active;
  active.reserve(all.size());
  for (const auto& d : all)
    if (!d.fixed)
      active.push_back(d);
  return active;
}

std::vector<ParamDescriptor> load_default_param_descriptors() {
  std::string config_dir = UVA_PADOVA_CONFIG_DIR;
  std::string default_path = config_dir + "/default_params.json";
  return load_param_descriptors(default_path);
}

} // namespace uva_padova
