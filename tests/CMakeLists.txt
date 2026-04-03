set(FAST_FSS_TEST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest/googletest
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/wideint/include)

add_library(fast_fss_gtest_main STATIC
    ../third_party/googletest/googletest/src/gtest-all.cc
    ../third_party/googletest/googletest/src/gtest_main.cc)
target_include_directories(fast_fss_gtest_main PUBLIC ${FAST_FSS_TEST_INCLUDE_DIRS})

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

fast_fss_add_cpu_eval_test(tests_cpu_dcf_eval cpu/dcf/cpu_dcf_eval.cpp)
fast_fss_add_cpu_eval_test(tests_cpu_dpf_eval cpu/dpf/cpu_dpf_eval.cpp)
fast_fss_add_cpu_eval_test(tests_cpu_dpf_eval_all cpu/dpf/cpu_dpf_eval_all.cpp)
fast_fss_add_cpu_eval_test(tests_cpu_dpf_eval_multi cpu/dpf/cpu_dpf_eval_multi.cpp)
fast_fss_add_cpu_eval_test(tests_cpu_mic_eval cpu/mic/cpu_mic_eval.cpp)
fast_fss_add_cpu_eval_test(tests_cpu_grotto_eq_eval cpu/grotto/cpu_grotto_eq_eval.cpp)
fast_fss_add_cpu_eval_test(tests_cpu_grotto_eval cpu/grotto/cpu_grotto_eval.cpp)
fast_fss_add_cpu_eval_test(tests_cpu_grotto_mic_eval cpu/grotto/cpu_grotto_mic_eval.cpp)
fast_fss_add_cpu_eval_test(tests_cpu_grotto_interval_lut_eval cpu/grotto/cpu_grotto_interval_lut_eval.cpp)
fast_fss_add_cpu_eval_test(tests_cpu_ottt_lut_eval cpu/ottt/cpu_ottt_lut_eval.cpp)

if(CMAKE_CUDA_COMPILER)
    fast_fss_add_cuda_eval_test(tests_cuda_dcf_eval cuda/dcf/cuda_dcf_eval.cu)
    fast_fss_add_cuda_eval_test(tests_cuda_dpf_eval cuda/dpf/cuda_dpf_eval.cu)
    fast_fss_add_cuda_eval_test(tests_cuda_dpf_eval_all cuda/dpf/cuda_dpf_eval_all.cu)
    fast_fss_add_cuda_eval_test(tests_cuda_dpf_eval_multi cuda/dpf/cuda_dpf_eval_multi.cu)
    fast_fss_add_cuda_eval_test(tests_cuda_mic_eval cuda/mic/cuda_mic_eval.cu)
    fast_fss_add_cuda_eval_test(tests_cuda_grotto_eq_eval cuda/grotto/cuda_grotto_eq_eval.cu)
    fast_fss_add_cuda_eval_test(tests_cuda_grotto_eval cuda/grotto/cuda_grotto_eval.cu)
    fast_fss_add_cuda_eval_test(tests_cuda_grotto_mic_eval cuda/grotto/cuda_grotto_mic_eval.cu)
    fast_fss_add_cuda_eval_test(tests_cuda_grotto_interval_lut_eval cuda/grotto/cuda_grotto_interval_lut_eval.cu)
    fast_fss_add_cuda_eval_test(tests_cuda_ottt_lut_eval cuda/ottt/cuda_ottt_lut_eval.cu)
endif()





