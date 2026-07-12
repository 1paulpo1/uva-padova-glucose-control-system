add_executable(cross_validate
  cross_validate.cpp
  virtual_patient.cpp
  virtual_patient.hpp
)
target_link_libraries(cross_validate PRIVATE uva_padova_id)
target_include_directories(cross_validate PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(cross_validate PRIVATE
  UVA_PADOVA_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config"
)

add_executable(diagnose_rk4_error
  diagnose_rk4_error.cpp
  virtual_patient.cpp
  virtual_patient.hpp
)
target_link_libraries(diagnose_rk4_error PRIVATE uva_padova_id)
target_include_directories(diagnose_rk4_error PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  /opt/homebrew/include  # brew Boost (header-only odeint)
)
target_compile_definitions(diagnose_rk4_error PRIVATE
  UVA_PADOVA_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config"
  CROSSVAL_OUTPUT_DIR="${CMAKE_SOURCE_DIR}/crossval/output"
)
