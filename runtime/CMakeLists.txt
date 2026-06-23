cmake_minimum_required(VERSION 3.16)
project(gemma_npu_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gemma-npu-block-runner
  src/main.cpp
  src/block_runner.cpp
)

target_include_directories(gemma-npu-block-runner PRIVATE include src)
target_link_libraries(gemma-npu-block-runner PRIVATE dl)

add_library(gemma_reference_adapter SHARED
  adapters/reference_copy_adapter.cpp
)

target_include_directories(gemma_reference_adapter PRIVATE include)
