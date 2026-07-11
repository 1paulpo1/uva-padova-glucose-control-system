set(VALIDATION_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(VALIDATION_FIGURES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/figures")

add_executable(run_scenarios
  run_scenarios.cpp
)

target_link_libraries(run_scenarios
  PRIVATE
    uva_padova_core
)

target_include_directories(run_scenarios
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(run_scenarios
  PRIVATE
    VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR}"
    VALIDATION_FIGURES_DIR="${VALIDATION_FIGURES_DIR}"
)

add_executable(test_real_data
  test_real_data.cpp
  csv_reader.cpp
  csv_reader.hpp
)

target_link_libraries(test_real_data
  PRIVATE
    uva_padova_id
)

target_include_directories(test_real_data
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
