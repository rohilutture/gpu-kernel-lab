cmake_minimum_required(VERSION 3.24)
# RTX 5090 = Blackwell sm_120 (needs CUDA >= 12.8). Must be set BEFORE project().
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "")
project(kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(bench
  bench/bench.cu src/softmax.cu src/rmsnorm.cu src/attention.cu src/gemv_int4.cu)
target_include_directories(bench PRIVATE include)
# -lineinfo lets Nsight Compute map SASS back to source lines; no --use_fast_math so numerics stay honest
target_compile_options(bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas=-v>)
