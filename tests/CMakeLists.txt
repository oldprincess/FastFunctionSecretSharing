set(FAST_FSS_TEST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest/googletest
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/wideint/include)

add_library(fast_fss_gtest_main STATIC
    ../third_party/googletest/googletest/src/gtest-all.cc
    ../third_party/googletest/googletest/src/gtest_main.cc)
target_include_directories(fast_fss_gtest_main PUBLIC ${FAST_FSS_TEST_INCLUDE_DIRS})
target_compile_features(fast_fss_gtest_main PRIVATE cxx_std_23)

function(fast_fss_add_cpu_eval_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE FastFss fast_fss_gtest_main)
    target_include_directories(${target} PRIVATE ${FAST_FSS_TEST_INCLUDE_DIRS})
    target_compile_features(${target} PRIVATE cxx_std_20)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:FastFss> $<TARGET_FILE_DIR:${target}>)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

function(fast_fss_add_cuda_eval_test target)
    add_executable(${target} ${ARGN})
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${target} PRIVATE FastFss fast_fss_gtest_main CUDA::cudart)
    target_include_directories(${target} PRIVATE ${FAST_FSS_TEST_INCLUDE_DIRS})
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE             
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)

    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:FastFss> $<TARGET_FILE_DIR:${target}>)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

fast_fss_add_cpu_eval_test(dcf_eval_cpu dcf/dcf_eval_cpu.cpp)

fast_fss_add_cpu_eval_test(dpf_eval_all_cpu dpf/dpf_eval_all_cpu.cpp)
fast_fss_add_cpu_eval_test(dpf_eval_cpu dpf/dpf_eval_cpu.cpp)
fast_fss_add_cpu_eval_test(dpf_eval_multi_cpu dpf/dpf_eval_multi_cpu.cpp)

fast_fss_add_cpu_eval_test(grotto_eq_eval_cpu grotto/grotto_eq_eval_cpu.cpp)
fast_fss_add_cpu_eval_test(grotto_eval_cpu grotto/grotto_eval_cpu.cpp)
fast_fss_add_cpu_eval_test(grotto_interval_lut_eval_cpu grotto/grotto_interval_lut_eval_cpu.cpp)
fast_fss_add_cpu_eval_test(grotto_mic_eval_cpu grotto/grotto_mic_eval_cpu.cpp)

fast_fss_add_cpu_eval_test(mic_eval_cpu mic/mic_eval_cpu.cpp)

fast_fss_add_cpu_eval_test(ottt_eval_cpu ottt/ottt_eval_cpu.cpp)

fast_fss_add_cpu_eval_test(spline_eval_cpu spline/spline_eval_cpu.cpp)

if(FAST_FSS_ENABLE_CUDA AND CMAKE_CUDA_COMPILER AND TARGET CUDA::cudart)
    fast_fss_add_cuda_eval_test(dcf_eval_cuda dcf/dcf_eval_cuda.cu)
    set_target_properties(dcf_eval_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

    fast_fss_add_cuda_eval_test(dpf_eval_all_cuda dpf/dpf_eval_all_cuda.cu)
    set_target_properties(dpf_eval_all_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(dpf_eval_cuda dpf/dpf_eval_cuda.cu)
    set_target_properties(dpf_eval_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(dpf_eval_multi_cuda dpf/dpf_eval_multi_cuda.cu)
    set_target_properties(dpf_eval_multi_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

    fast_fss_add_cuda_eval_test(grotto_eq_eval_cuda grotto/grotto_eq_eval_cuda.cu)
    set_target_properties(grotto_eq_eval_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(grotto_eval_cuda grotto/grotto_eval_cuda.cu)
    set_target_properties(grotto_eval_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(grotto_interval_lut_eval_cuda grotto/grotto_interval_lut_eval_cuda.cu)
    set_target_properties(grotto_interval_lut_eval_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    fast_fss_add_cuda_eval_test(grotto_mic_eval_cuda grotto/grotto_mic_eval_cuda.cu)
    set_target_properties(grotto_mic_eval_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

    fast_fss_add_cuda_eval_test(mic_eval_cuda mic/mic_eval_cuda.cu)
    set_target_properties(mic_eval_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

    fast_fss_add_cuda_eval_test(ottt_eval_cuda ottt/ottt_eval_cuda.cu)
    set_target_properties(ottt_eval_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

    fast_fss_add_cuda_eval_test(spline_eval_cuda spline/spline_eval_cuda.cu)
    set_target_properties(spline_eval_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
endif()





