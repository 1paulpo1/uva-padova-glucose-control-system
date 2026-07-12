# ===========================================================================
# Unit tests for core library (with MockSimulator)
# ===========================================================================
add_executable(unit_core
  unit/adjustments_test.cpp
  unit/prior_cost_test.cpp
  unit/observation_cost_test.cpp
)
target_link_libraries(unit_core PRIVATE uva_padova_core Ceres::ceres GTest::gtest GTest::gtest_main)
target_include_directories(unit_core PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_options(unit_core PRIVATE -Wl,-rpath,/opt/anaconda3/lib)

# ===========================================================================
# Integration tests for identification library (with real ODE)
# ===========================================================================
add_executable(unit_id
  integration/observation_cost_integration_test.cpp
)
target_link_libraries(unit_id PRIVATE uva_padova_id GTest::gtest GTest::gtest_main)
target_include_directories(unit_id PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_options(unit_id PRIVATE -Wl,-rpath,/opt/anaconda3/lib)

# ===========================================================================
# Unit tests for optimizer module (CMAESOptimizer, LMOptimizer, Pipeline)
# ===========================================================================
add_executable(test_optimizer
  unit/optimizer_test.cpp
)
target_link_libraries(test_optimizer PRIVATE uva_padova_id GTest::gtest GTest::gtest_main)
target_include_directories(test_optimizer PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_options(test_optimizer PRIVATE -Wl,-rpath,/opt/anaconda3/lib)

# ===========================================================================
# Register gtest-based tests with CTest
# ===========================================================================
include(GoogleTest)
gtest_discover_tests(unit_core)
gtest_discover_tests(unit_id)
gtest_discover_tests(test_optimizer)

# ===========================================================================
# Existing tests — migrated to gtest
# ===========================================================================
add_executable(test_odesystem test_odesystem.cpp)
target_link_libraries(test_odesystem PRIVATE uva_padova_core GTest::gtest GTest::gtest_main)
target_include_directories(test_odesystem PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_options(test_odesystem PRIVATE -Wl,-rpath,/opt/anaconda3/lib)

add_executable(test_identification integration/identification_pipeline_test.cpp)
target_link_libraries(test_identification PRIVATE uva_padova_id GTest::gtest GTest::gtest_main)
target_include_directories(test_identification PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests)
target_link_options(test_identification PRIVATE -Wl,-rpath,/opt/anaconda3/lib)

gtest_discover_tests(test_odesystem)
gtest_discover_tests(test_identification)

add_executable(exp_identifiability exp_identifiability.cpp)
target_link_libraries(exp_identifiability PRIVATE uva_padova_id)
target_include_directories(exp_identifiability PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(debug_fim debug_fim.cpp)
target_link_libraries(debug_fim PRIVATE uva_padova_id)
target_include_directories(debug_fim PRIVATE ${CMAKE_SOURCE_DIR}/include)
