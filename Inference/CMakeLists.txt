add_executable(Inference main.cpp)

# -------------------------
# vcpkg packages
# -------------------------
find_package(OpenCV CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(log4cxx CONFIG REQUIRED)

# -------------------------
# CUDA (Native cross-platform detection)
# -------------------------
find_package(CUDAToolkit REQUIRED)

# -------------------------
# TensorRT
# -------------------------
set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
if(WIN32)
    add_library(nvinfer SHARED IMPORTED)
    set_target_properties(nvinfer PROPERTIES
        IMPORTED_LOCATION "${TENSORRT_ROOT}/lib/nvinfer.dll"
        IMPORTED_IMPLIB "${TENSORRT_ROOT}/lib/nvinfer.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_ROOT}/include"
    )
    add_library(nvonnxparser SHARED IMPORTED)
    set_target_properties(nvonnxparser PROPERTIES
        IMPORTED_LOCATION "${TENSORRT_ROOT}/lib/nvonnxparser.dll"
        IMPORTED_IMPLIB "${TENSORRT_ROOT}/lib/nvonnxparser.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_ROOT}/include"
    )
else()
    add_library(nvinfer SHARED IMPORTED)
    set_target_properties(nvinfer PROPERTIES
        IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libnvinfer.so"
        INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
    )
    add_library(nvonnxparser SHARED IMPORTED)
    set_target_properties(nvonnxparser PROPERTIES
        IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libnvonnxparser.so"
        INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
    )
endif()

# -------------------------
# Copy Assets (IMPORTANT)
# -------------------------
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/Assets)
add_custom_command(TARGET Inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
)
add_custom_command(TARGET Inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../Assets/net.trt
        ${OUTPUT_DIR}/net.trt
)
add_custom_command(TARGET Inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../Assets/test_signal.raw
        ${OUTPUT_DIR}/test_signal.raw
)

# -------------------------
# Link Libraries
# -------------------------
target_link_libraries(Inference PRIVATE
    fmt::fmt
    nlohmann_json::nlohmann_json
    log4cxx

    ${OpenCV_LIBS}

    nvinfer
    nvonnxparser

    CUDA::cudart
)