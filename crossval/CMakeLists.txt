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

add_executable(topology_explorer
  topology_explorer.cpp
  virtual_patient.cpp
  virtual_patient.hpp
)
target_link_libraries(topology_explorer PRIVATE uva_padova_id)
target_include_directories(topology_explorer PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(topology_explorer PRIVATE
  UVA_PADOVA_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config"
)
