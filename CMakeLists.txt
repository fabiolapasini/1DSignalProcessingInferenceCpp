cmake_minimum_required(VERSION 3.8)

project(1DSignalProcessingInferenceC++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build tests" ON)

if (BUILD_TESTS)
    enable_testing()
    list(APPEND VCPKG_MANIFEST_FEATURES "test")
endif()

add_subdirectory(Inference)