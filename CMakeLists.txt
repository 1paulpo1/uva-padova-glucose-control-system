cmake_minimum_required(VERSION 3.20)
project(uva_padova LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 REQUIRED)

# gflags: allow override via gflags_DIR; Ceres will find it transitively.
# No hard-coded Homebrew paths — works on Linux, macOS (brew --prefix), and CI.
find_package(gflags QUIET)
if(NOT gflags_FOUND)
  # Try common Homebrew locations without hard-coding version
  foreach(_brew_prefix "/opt/homebrew" "/usr/local" "/opt/local")
    if(EXISTS "${_brew_prefix}/include/gflags/gflags.h")
      list(APPEND CMAKE_PREFIX_PATH "${_brew_prefix}")
    endif()
  endforeach()
  find_package(gflags REQUIRED)
endif()
# Create alias so Ceres' interface dependency finds `gflags` target
if(NOT TARGET gflags)
  foreach(_candidate gflags_shared gflags_static gflags::gflags_shared gflags::gflags_static)
    if(TARGET ${_candidate})
      add_library(gflags ALIAS ${_candidate})
      break()
    endif()
  endforeach()
endif()
find_package(Ceres REQUIRED)

find_package(GTest REQUIRED)
enable_testing()

find_package(libcmaes REQUIRED)

# nlohmann/json (header-only, zero-dependency)
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
  )
  set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

find_package(matplotplusplus QUIET)
if(matplotplusplus_FOUND)
  set(HAS_MATPLOTPLUS ON)
  message(STATUS "matplotplusplus found")
else()
  set(HAS_MATPLOTPLUS OFF)
  message(STATUS "matplotplusplus NOT found - validation plots disabled")
endif()

# Core model library (no heavy deps)
add_library(uva_padova_core
  src/types.cpp
  src/odesystem.cpp
  src/integrals.cpp
)

target_include_directories(uva_padova_core
  PUBLIC
    include
    ${EIGEN3_INCLUDE_DIR}
)

# unscented (UKF) library — optional, see https://github.com/kam3k/unscented
# Portable: check via CMAKE_PREFIX_PATH instead of hard-coded /opt/homebrew or /mingw64
find_path(UNSCENTED_INCLUDE_DIR NAMES unscented/unscented.h PATH_SUFFIXES include)
if(UNSCENTED_INCLUDE_DIR)
  target_include_directories(uva_padova_core PUBLIC ${UNSCENTED_INCLUDE_DIR})
endif()

target_link_libraries(uva_padova_core
  PUBLIC
    Eigen3::Eigen
)

# Identification library (Ceres, libcmaes)
add_library(uva_padova_id
  src/optimizer.cpp
  src/diagnostics.cpp
)

target_include_directories(uva_padova_id
  PUBLIC
    include
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(uva_padova_id
  PUBLIC
    uva_padova_core
    ${CERES_LIBRARIES}
    libcmaes::cmaes
    nlohmann_json::nlohmann_json
)

target_sources(uva_padova_id
  PRIVATE
    src/config_loader.cpp
)

target_compile_definitions(uva_padova_id
  PRIVATE
    UVA_PADOVA_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config"
)

# Online adaptation library (CKF, RTS smoother, meal priors)
add_library(uva_padova_online INTERFACE)
target_include_directories(uva_padova_online
  INTERFACE
    include
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(uva_padova_online
  INTERFACE
    uva_padova_id
)
target_sources(uva_padova_online
  INTERFACE
    include/uva_padova/augmented_state.hpp
    include/uva_padova/cubature_kalman_filter.hpp
    include/uva_padova/system_model.hpp
    include/uva_padova/measurement_model.hpp
    include/uva_padova/regime_modulated_q.hpp
    include/uva_padova/rts_smoother.hpp
    include/uva_padova/meal_prior_store.hpp
    include/uva_padova/online_adaptation.hpp
)

target_compile_options(uva_padova_core PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(uva_padova_id PRIVATE -Wall -Wextra -Wpedantic)

# Tests
add_subdirectory(tests)

# Real-world validation (CGM data from T1DEXI / OhioT1DM)
add_subdirectory(real_world_validation)

# Synthetic identifiability study (virtual patients from the simulator itself)
add_subdirectory(synthetic_study)
