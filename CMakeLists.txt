cmake_minimum_required(VERSION 2.8)
project(visualodometry)


set(CMAKE_BUILD_TYPE "Debug")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# 寻找OpenCV库
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
include_directories()

# Eigen
include_directories("/usr/include/eigen3")
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS}
                    ${G2O_INCLUDE_DIRS}
                    ${Sophus_INCLUDE_DIRS}
                    ${Pangolin_INCLUDE_DIRS})


add_executable(visualodometry VisualOdometry2D.cpp)
target_link_libraries(visualodometry ${OpenCV_LIBS}
${Pangolin_LIBRARIES})

