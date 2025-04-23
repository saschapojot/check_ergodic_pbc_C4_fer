cmake_minimum_required(VERSION 3.28)
project(pbc_C3_fer LANGUAGES CXX)


message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")
# Find Python
find_package(Python3 REQUIRED COMPONENTS Development)
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Boost
set(Boost_NO_SYSTEM_PATHS ON)
set(BOOST_ROOT "$ENV{HOME}/cppLocal/boostBuild")
find_package(Boost REQUIRED COMPONENTS python numpy filesystem)

# Armadillo configuration
find_path(ARMADILLO_INCLUDE_DIR armadillo HINTS "/home/adada/cppLocal/armadillo_intel/include")
find_library(ARMADILLO_LIBRARY armadillo HINTS "/home/adada/cppLocal/armadillo_intel/lib")


# Add executable
add_executable(run_mc
        run_mc.cpp
        mc_subroutine/mc_read_load_compute.cpp
)


# Set specific optimization for critical files
set_source_files_properties(
        mc_subroutine/mc_read_load_compute.cpp
        run_mc.cpp
        PROPERTIES COMPILE_OPTIONS "-O3"
)


# Include directories
target_include_directories(run_mc PRIVATE

        ${ARMADILLO_INCLUDE_DIR}
        ${Python3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(run_mc PRIVATE
        pthread
        m
        ${ARMADILLO_LIBRARY}
        Boost::filesystem
        Boost::python
        Boost::numpy
        Python3::Python
)