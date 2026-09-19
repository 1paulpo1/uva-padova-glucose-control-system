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

# ===========================================================================
# Integration tests for identification library (with real ODE)
# ===========================================================================
add_executable(unit_id
  integration/observation_cost_integration_test.cpp
)
target_link_libraries(unit_id PRIVATE uva_padova_id GTest::gtest GTest::gtest_main)
target_include_directories(unit_id PRIVATE ${CMAKE_SOURCE_DIR}/tests)

# ===========================================================================
# Unit tests for optimizer module (CMAESOptimizer, LMOptimizer, Pipeline)
# ===========================================================================
add_executable(test_optimizer
  unit/optimizer_test.cpp
)
target_link_libraries(test_optimizer PRIVATE uva_padova_id GTest::gtest GTest::gtest_main)
target_include_directories(test_optimizer PRIVATE ${CMAKE_SOURCE_DIR}/tests)

# ===========================================================================
# Register gtest-based tests with CTest
# ===========================================================================
# ===========================================================================
# CKF online adaptation tests
# ===========================================================================
add_executable(test_ckf
  unit/ckf_test.cpp
)
target_link_libraries(test_ckf PRIVATE uva_padova_online GTest::gtest GTest::gtest_main)
target_include_directories(test_ckf PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Note: no hard-coded rpath — use CMAKE_INSTALL_RPATH or conda activation instead
include(GoogleTest)
gtest_discover_tests(unit_core)
gtest_discover_tests(test_ckf)
gtest_discover_tests(unit_id)
gtest_discover_tests(test_optimizer)

# ===========================================================================
# Existing tests — migrated to gtest
# ===========================================================================
add_executable(test_odesystem test_odesystem.cpp)
target_link_libraries(test_odesystem PRIVATE uva_padova_core GTest::gtest GTest::gtest_main)
target_include_directories(test_odesystem PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_identification integration/identification_pipeline_test.cpp)
target_link_libraries(test_identification PRIVATE uva_padova_id GTest::gtest GTest::gtest_main)
target_include_directories(test_identification PRIVATE ${CMAKE_SOURCE_DIR}/include)

gtest_discover_tests(test_odesystem)
gtest_discover_tests(test_identification)

add_executable(exp_identifiability exp_identifiability.cpp)
target_link_libraries(exp_identifiability PRIVATE uva_padova_id)
target_include_directories(exp_identifiability PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(debug_fim debug_fim.cpp)
target_link_libraries(debug_fim PRIVATE uva_padova_id)
target_include_directories(debug_fim PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ===========================================================================
# Recovery benchmark tests
# ===========================================================================
add_executable(test_recovery
  recovery/recovery_benchmark_test.cpp
)
target_link_libraries(test_recovery PRIVATE uva_padova_id GTest::gtest GTest::gtest_main)
target_include_directories(test_recovery PRIVATE ${CMAKE_SOURCE_DIR}/include)

gtest_discover_tests(test_recovery)
