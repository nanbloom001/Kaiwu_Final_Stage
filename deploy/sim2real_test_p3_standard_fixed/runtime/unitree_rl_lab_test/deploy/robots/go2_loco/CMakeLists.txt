cmake_minimum_required(VERSION 3.12)
project(go2_loco_controller)

# C++17（std::filesystem / std::string_view / std::clamp / inline 变量；GCC9 默认 gnu++14 不够）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# onnxruntime 根目录：Jetson(aarch64) 用 1.19.2；可在命令行用 -DONNXRUNTIME_ROOT=... 覆盖
set(ONNXRUNTIME_ROOT "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-1.19.2"
    CACHE PATH "onnxruntime 解压根目录（含 include/ 与 lib/）")
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "找不到 onnxruntime: ${ONNXRUNTIME_ROOT}（用 -DONNXRUNTIME_ROOT=<解压目录> 指定）")
endif()
find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "在 ${ONNXRUNTIME_ROOT}/lib 找不到 libonnxruntime.so")
endif()

# RealSense（D435i 深度）：默认开；装了 librealsense2-dev 即可。关闭用 -DUSE_REALSENSE=OFF
option(USE_REALSENSE "用 librealsense2 直连 D435i 取深度" ON)
if(USE_REALSENSE)
  find_library(REALSENSE2_LIB realsense2)
  find_path(REALSENSE2_INC librealsense2/rs.hpp)
  if(REALSENSE2_LIB AND REALSENSE2_INC)
    message(STATUS "RealSense: ${REALSENSE2_LIB}")
    add_compile_definitions(USE_REALSENSE)
  else()
    message(WARNING "未找到 librealsense2（装 librealsense2-dev）；本次构建关闭 USE_REALSENSE，仅 ConstantDepth 可用。")
    set(USE_REALSENSE OFF)
  endif()
endif()


find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${ONNXRUNTIME_ROOT}/include
  ${PROJECT_SOURCE_DIR}/../../include/
)
if(USE_REALSENSE)
  include_directories(${REALSENSE2_INC})
endif()

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${ONNXRUNTIME_LIB}
)
if(USE_REALSENSE)
  link_libraries(${REALSENSE2_LIB})
endif()

set(ADD_SRC_LIST
  src/State_VisionLoco.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_loco_ctrl main.cpp)
