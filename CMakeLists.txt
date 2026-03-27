cmake_minimum_required(VERSION 3.21)

project(FastFunctionSecretSharing LANGUAGES CXX)

include(CTest)
include(CheckLanguage)

option(FASTFSS_ENABLE_CUDA "Build CUDA targets when a CUDA toolchain is available." ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(FASTFSS_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FASTFSS_PUBLIC_INCLUDE_DIRS
    ${FASTFSS_ROOT_DIR}/include
    ${FASTFSS_ROOT_DIR}/src
    ${FASTFSS_ROOT_DIR}/third_party/wideint/include)

set(FASTFSS_CPU_SOURCES
    src/cpu/config.cpp
    src/cpu/dcf.cpp
    src/cpu/dpf.cpp
    src/cpu/grotto.cpp
    src/cpu/mic.cpp
    src/cpu/ottt.cpp
    src/cpu/prng.cpp)

set(FASTFSS_CPU_TEST_SOURCES
    test/cpu/aes.cpp
    test/cpu/dcf.cpp
    test/cpu/dpf.cpp
    test/cpu/grotto.cpp
    test/cpu/mic.cpp
    test/cpu/ottt.cpp
    test/cpu/prng.cpp)

set(FASTFSS_CUDA_SOURCES
    src/cuda/config.cpp
    src/cuda/dcf.cu
    src/cuda/dpf.cu
    src/cuda/grotto.cu
    src/cuda/mic.cu
    src/cuda/ottt.cu
    src/cuda/prng.cu)

set(FASTFSS_CUDA_TEST_SOURCES
    test/cuda/aes.cu
    test/cuda/dcf.cu
    test/cuda/dpf.cu
    test/cuda/grotto.cu
    test/cuda/mic.cu
    test/cuda/ottt.cu
    test/cuda/prng.cu)

function(fastfss_apply_common target_name)
    target_compile_features(${target_name} PUBLIC cxx_std_17)
    target_include_directories(${target_name}
        PUBLIC
            ${FASTFSS_PUBLIC_INCLUDE_DIRS})

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target_name}
            PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:-maes>)
    endif()
endfunction()

add_library(FastFssCpu ${FASTFSS_CPU_SOURCES})
add_library(FastFss::Cpu ALIAS FastFssCpu)
fastfss_apply_common(FastFssCpu)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(FastFssCpu PUBLIC OpenMP::OpenMP_CXX)
endif()

set(FASTFSS_HAVE_CUDA OFF)
if(FASTFSS_ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        find_package(CUDAToolkit REQUIRED)

        add_library(FastFssCuda ${FASTFSS_CUDA_SOURCES})
        add_library(FastFss::Cuda ALIAS FastFssCuda)
        fastfss_apply_common(FastFssCuda)
        target_link_libraries(FastFssCuda PUBLIC CUDA::cudart)
        target_compile_options(FastFssCuda
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        set_target_properties(FastFssCuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set(FASTFSS_HAVE_CUDA ON)
    endif()
endif()

if(BUILD_TESTING)
    if(NOT EXISTS ${FASTFSS_ROOT_DIR}/third_party/googletest/CMakeLists.txt)
        message(FATAL_ERROR "googletest submodule is missing. Run: git submodule update --init --recursive")
    endif()

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
    include(GoogleTest)

    function(fastfss_add_cpu_test source_file)
        get_filename_component(test_name ${source_file} NAME_WE)
        set(target_name cpu_${test_name}_gtest)
        add_executable(${target_name} ${source_file})
        if(test_name STREQUAL "aes")
            target_link_libraries(${target_name}
                PRIVATE
                    GTest::gtest_main)
        else()
            target_link_libraries(${target_name}
                PRIVATE
                    FastFssCpu
                    GTest::gtest_main)
        endif()
        target_include_directories(${target_name}
            PRIVATE
                ${FASTFSS_PUBLIC_INCLUDE_DIRS}
                ${FASTFSS_ROOT_DIR}/test/cpu)
        gtest_discover_tests(${target_name})
    endfunction()

    foreach(source_file IN LISTS FASTFSS_CPU_TEST_SOURCES)
        fastfss_add_cpu_test(${source_file})
    endforeach()

    if(FASTFSS_HAVE_CUDA)
        function(fastfss_add_cuda_test source_file)
            get_filename_component(test_name ${source_file} NAME_WE)
            set(target_name cuda_${test_name}_gtest)
            add_executable(${target_name} ${source_file})
            if(test_name STREQUAL "aes")
                target_link_libraries(${target_name}
                    PRIVATE
                        CUDA::cudart
                        GTest::gtest_main)
            else()
                target_link_libraries(${target_name}
                    PRIVATE
                        FastFssCuda
                        CUDA::cudart
                        GTest::gtest_main)
            endif()
            target_include_directories(${target_name}
                PRIVATE
                    ${FASTFSS_PUBLIC_INCLUDE_DIRS}
                    ${FASTFSS_ROOT_DIR}/test/cuda)
            target_compile_options(${target_name}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
            set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
            gtest_discover_tests(${target_name})
        endfunction()

        foreach(source_file IN LISTS FASTFSS_CUDA_TEST_SOURCES)
            fastfss_add_cuda_test(${source_file})
        endforeach()
    endif()
endif()
