set(VALIDATION_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(VALIDATION_FIGURES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/figures")

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

target_compile_definitions(test_real_data
  PRIVATE
    UVA_PADOVA_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config"
    VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR}"
    VALIDATION_FIGURES_DIR="${VALIDATION_FIGURES_DIR}"
)
