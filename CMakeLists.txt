cmake_minimum_required(VERSION 3.20)
project(uva_padova LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 REQUIRED)

# Use shared gflags to avoid duplicate static registrations
if(EXISTS /opt/homebrew/Cellar/gflags/2.3.0/lib/cmake/gflags)
  set(gflags_DIR /opt/homebrew/Cellar/gflags/2.3.0/lib/cmake/gflags CACHE PATH "Path to gflags CMake config")
endif()
find_package(gflags REQUIRED)
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
if(EXISTS /opt/homebrew/include/unscented)
  target_include_directories(uva_padova_core PUBLIC /opt/homebrew/include/unscented)
elseif(EXISTS /mingw64/include/unscented)
  target_include_directories(uva_padova_core PUBLIC /mingw64/include/unscented)
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
  PRIVATE
    /opt/homebrew/include
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

target_compile_options(uva_padova_core PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(uva_padova_id PRIVATE -Wall -Wextra -Wpedantic)

# Tests
add_subdirectory(tests)

# Validation (CSV output, plotted via Python)
add_subdirectory(validation)

# Cross-validation
add_subdirectory(crossval)
