cmake_minimum_required(VERSION 3.18)

include(CheckLanguage)
PROJECT(VBD LANGUAGES CXX CUDA)

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(CMAKE_VERBOSE_MAKEFILE OFF)
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
file(GLOB_RECURSE HEADERS "src/*.h")

# remove main so we can relink later
list(REMOVE_ITEM CPP_SOURCES "src/main.cpp")

add_library(CUDA_Lib STATIC ${CUDA_SOURCES})
add_library(CPP_Lib STATIC ${CPP_SOURCES})

find_package(Eigen3 REQUIRED)
target_link_libraries(CUDA_Lib PRIVATE Eigen3::Eigen)
target_link_libraries(CPP_Lib PRIVATE Eigen3::Eigen)

find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(CUDA_Lib PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(CPP_Lib PUBLIC nlohmann_json::nlohmann_json)

find_package(VTK REQUIRED)
target_link_libraries(CPP_Lib PRIVATE ${VTK_LIBRARIES})

# Link both into the main executable
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA_Lib CPP_Lib)
