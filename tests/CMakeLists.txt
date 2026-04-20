set(FAST_FSS_TEST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest/googletest
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/wideint/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xtensor/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xtl/include)

add_library(fast_fss_gtest_main STATIC
    ../third_party/googletest/googletest/src/gtest-all.cc
    ../third_party/googletest/googletest/src/gtest_main.cc)
target_include_directories(fast_fss_gtest_main PUBLIC ${FAST_FSS_TEST_INCLUDE_DIRS})
target_compile_features(fast_fss_gtest_main PRIVATE cxx_std_23)

function(fast_fss_add_cpu_eval_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE FastFss fast_fss_gtest_main)
    target_include_directories(${target} PRIVATE ${FAST_FSS_TEST_INCLUDE_DIRS})
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:FastFss> $<TARGET_FILE_DIR:${target}>)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

function(fast_fss_add_cuda_eval_test target)
    add_executable(${target} ${ARGN})
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${target} PRIVATE FastFss fast_fss_gtest_main CUDA::cudart)
    target_include_directories(${target} PRIVATE ${FAST_FSS_TEST_INCLUDE_DIRS})
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:FastFss> $<TARGET_FILE_DIR:${target}>)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

# DCF eval tests under tests/dcf/
fast_fss_add_cpu_eval_test(tests_cpu_dcf_eval dcf/dcf_eval_cpu.cpp)
target_compile_features(tests_cpu_dcf_eval PRIVATE cxx_std_20)
fast_fss_add_cpu_eval_test(tests_cpu_dpf_eval dpf/dpf_eval_cpu.cpp)
target_compile_features(tests_cpu_dpf_eval PRIVATE cxx_std_20)
fast_fss_add_cpu_eval_test(tests_cpu_dpf_eval_all dpf/dpf_eval_all_cpu.cpp)
target_compile_features(tests_cpu_dpf_eval_all PRIVATE cxx_std_20)
fast_fss_add_cpu_eval_test(tests_cpu_dpf_eval_multi dpf/dpf_eval_multi_cpu.cpp)
target_compile_features(tests_cpu_dpf_eval_multi PRIVATE cxx_std_20)

if(FAST_FSS_ENABLE_CUDA AND CMAKE_CUDA_COMPILER AND TARGET CUDA::cudart)
    fast_fss_add_cuda_eval_test(tests_cuda_dcf_eval dcf/dcf_eval_cuda.cu)
    target_compile_features(tests_cuda_dcf_eval PRIVATE cxx_std_20)
    set_target_properties(tests_cuda_dcf_eval PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(tests_cuda_dpf_eval dpf/dpf_eval_cuda.cu)
    target_compile_features(tests_cuda_dpf_eval PRIVATE cxx_std_20)
    set_target_properties(tests_cuda_dpf_eval PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(tests_cuda_dpf_eval_all dpf/dpf_eval_all_cuda.cu)
    target_compile_features(tests_cuda_dpf_eval_all PRIVATE cxx_std_20)
    set_target_properties(tests_cuda_dpf_eval_all PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(tests_cuda_dpf_eval_multi dpf/dpf_eval_multi_cuda.cu)
    target_compile_features(tests_cuda_dpf_eval_multi PRIVATE cxx_std_20)
    set_target_properties(tests_cuda_dpf_eval_multi PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
endif()

fast_fss_add_cpu_eval_test(tests_cpu_mic_eval mic/mic_eval_cpu.cpp)
target_compile_features(tests_cpu_mic_eval PRIVATE cxx_std_20)
fast_fss_add_cpu_eval_test(tests_cpu_ottt_eval ottt/ottt_eval_cpu.cpp)
target_compile_features(tests_cpu_ottt_eval PRIVATE cxx_std_20)

fast_fss_add_cpu_eval_test(tests_cpu_grotto_eval grotto/grotto_eval_cpu.cpp)
target_compile_features(tests_cpu_grotto_eval PRIVATE cxx_std_20)
fast_fss_add_cpu_eval_test(tests_cpu_grotto_eq_eval grotto/grotto_eq_eval_cpu.cpp)
target_compile_features(tests_cpu_grotto_eq_eval PRIVATE cxx_std_20)
fast_fss_add_cpu_eval_test(tests_cpu_grotto_interval_lut_eval grotto/grotto_interval_lut_eval_cpu.cpp)
target_compile_features(tests_cpu_grotto_interval_lut_eval PRIVATE cxx_std_20)
fast_fss_add_cpu_eval_test(tests_cpu_grotto_mic_eval grotto/grotto_mic_eval_cpu.cpp)
target_compile_features(tests_cpu_grotto_mic_eval PRIVATE cxx_std_20)

if(FAST_FSS_ENABLE_CUDA AND CMAKE_CUDA_COMPILER AND TARGET CUDA::cudart)
    fast_fss_add_cuda_eval_test(tests_cuda_mic_eval mic/mic_eval_cuda.cu)
    target_compile_features(tests_cuda_mic_eval PRIVATE cxx_std_20)
    set_target_properties(tests_cuda_mic_eval PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(tests_cuda_ottt_eval ottt/ottt_eval_cuda.cu)
    target_compile_features(tests_cuda_ottt_eval PRIVATE cxx_std_20)
    set_target_properties(tests_cuda_ottt_eval PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(tests_cuda_grotto_eval grotto/grotto_eval_cuda.cu)
    target_compile_features(tests_cuda_grotto_eval PRIVATE cxx_std_20)
    set_target_properties(tests_cuda_grotto_eval PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(tests_cuda_grotto_eq_eval grotto/grotto_eq_eval_cuda.cu)
    target_compile_features(tests_cuda_grotto_eq_eval PRIVATE cxx_std_20)
    set_target_properties(tests_cuda_grotto_eq_eval PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(tests_cuda_grotto_interval_lut_eval grotto/grotto_interval_lut_eval_cuda.cu)
    target_compile_features(tests_cuda_grotto_interval_lut_eval PRIVATE cxx_std_20)
    set_target_properties(tests_cuda_grotto_interval_lut_eval PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(tests_cuda_grotto_mic_eval grotto/grotto_mic_eval_cuda.cu)
    target_compile_features(tests_cuda_grotto_mic_eval PRIVATE cxx_std_20)
    set_target_properties(tests_cuda_grotto_mic_eval PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
endif()





