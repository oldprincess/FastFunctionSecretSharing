cmake_minimum_required(VERSION 3.24)

project(FastFunctionSecretSharing LANGUAGES CXX)

include(CheckLanguage)

option(FAST_FSS_ENABLE_CUDA "Build CUDA targets when a CUDA toolchain is available." ON)
option(FAST_FSS_BUILD_TESTS "Build tests." OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(FAST_FSS_PUBLIC_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/wideint/include)

set(FAST_FSS_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dcf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dpf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/grotto.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ottt.cpp)

set(FAST_FSS_CPU_SOURCES 
    src/cpu/config.cpp
    src/cpu/dcf.cpp
    src/cpu/dpf.cpp
    src/cpu/grotto.cpp
    src/cpu/mic.cpp
    src/cpu/ottt.cpp
    src/cpu/prng.cpp)

set(FAST_FSS_CUDA_SOURCES 
    src/cuda/config.cpp
    src/cuda/dcf.cu
    src/cuda/dpf.cu
    src/cuda/grotto.cu
    src/cuda/mic.cu
    src/cuda/ottt.cu
    src/cuda/prng.cu)

add_library(FastFss SHARED ${FAST_FSS_SOURCES} ${FAST_FSS_CPU_SOURCES})

target_compile_features(FastFss PUBLIC cxx_std_20)
target_compile_definitions(FastFss PRIVATE FAST_FSS_BUILD)
target_include_directories(FastFss PUBLIC ${FAST_FSS_PUBLIC_INCLUDE_DIRS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(FastFss PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-maes>)
endif()
set_target_properties(FastFss PROPERTIES 
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

find_package(OpenMP)
target_link_libraries(FastFss PRIVATE OpenMP::OpenMP_CXX)

if(FAST_FSS_ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        find_package(CUDAToolkit REQUIRED)

        target_sources(FastFss PRIVATE ${FAST_FSS_CUDA_SOURCES})
        target_link_libraries(FastFss PRIVATE CUDA::cudart)
        target_compile_options(FastFss PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)

        set_target_properties(FastFss PROPERTIES
            CUDA_ARCHITECTURES native
            CUDA_VISIBILITY_PRESET hidden
            CUDA_SEPARABLE_COMPILATION ON)
    endif()
endif()

if(FAST_FSS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
    add_subdirectory(tests)
endif()


