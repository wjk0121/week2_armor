cmake_minimum_required(VERSION 3.16)      # 最低 cmake 版本要求
project(rm_vision LANGUAGES CXX)          # 项目名称和语言(C++)

set(CMAKE_CXX_STANDARD 17)                # 使用 C++17 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)       # 标准为必需, 不允许降级
set(CMAKE_CXX_EXTENSIONS OFF)             # 禁止编译器 GNU 扩展

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)         # 默认 Release: 视频处理性能差距明显
endif()

#  导出编译数据库(compile_commands.json):
#  VS Code 的 IntelliSense 与 CMake Tools 靠它做代码补全/跳转, 建议保留
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#  OpenCV
find_package(OpenCV REQUIRED)

#  ONNX Runtime (ubuntu 下安装在 ~/onnxruntime)
set(ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS ${ONNXRUNTIME_ROOT}/include)
find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(rm_vision ${SOURCES})
target_include_directories(rm_vision PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(rm_vision
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY})

#  运行时自动找到 libonnxruntime.so, 免去每次设置 LD_LIBRARY_PATH
set_target_properties(rm_vision PROPERTIES BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
