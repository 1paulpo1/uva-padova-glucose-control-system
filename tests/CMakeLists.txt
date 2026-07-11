add_executable(test_odesystem test_odesystem.cpp)
target_link_libraries(test_odesystem PRIVATE uva_padova_core)
target_include_directories(test_odesystem PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_identification test_identification.cpp)
target_link_libraries(test_identification PRIVATE uva_padova_id)
target_include_directories(test_identification PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(exp_identifiability exp_identifiability.cpp)
target_link_libraries(exp_identifiability PRIVATE uva_padova_id)
target_include_directories(exp_identifiability PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(debug_fim debug_fim.cpp)
target_link_libraries(debug_fim PRIVATE uva_padova_id)
target_include_directories(debug_fim PRIVATE ${CMAKE_SOURCE_DIR}/include)
