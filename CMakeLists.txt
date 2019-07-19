cmake_minimum_required(VERSION 2.8.12)
project(sdot)

# Some configuration variables
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(WITH_OSX TRUE)
endif()

if(WITH_OSX)
    set(GEOGRAM_BUILD_DIR "${CMAKE_SOURCE_DIR}/lib/geogram/build/Darwin-clang-dynamic-Release")
else()
    set(GEOGRAM_BUILD_DIR "${CMAKE_SOURCE_DIR}/lib/geogram/build/Linux64-gcc-dynamic-Release")
endif()

# external libraries
## pybind11
set(PYTHON_EXECUTABLE "python3")
set(PYBIND11_PYTHON_VERSION 3.6)
add_subdirectory(lib/pybind11)
## NNRVD
include_directories(lib/NNRVD)
## laguerre
include_directories(lib/laguerre)
## CGAL
find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})
## Eigen
if(EXISTS "/usr/include/eigen3")
    set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
elseif(EXISTS "/usr/local/include/eigen3")
    set(EIGEN3_INCLUDE_DIR "/usr/local/include/eigen3")
endif()
include_directories(${EIGEN3_INCLUDE_DIR})
## geogram
execute_process(COMMAND sh -f configure.sh
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/geogram)
execute_process(COMMAND make -j4
                WORKING_DIRECTORY ${GEOGRAM_BUILD_DIR})
include_directories("${CMAKE_SOURCE_DIR}/lib/geogram/src/lib/")
link_directories("${GEOGRAM_BUILD_DIR}/lib")

# Python bindings
set(SOURCE_DIR "src/sdot/")
set(LIB_DIR "lib")
include_directories(${SOURCE_DIR})

# backends
## NNRVD
pybind11_add_module(backend_nnrvd SHARED ${SOURCE_DIR}/backend_nnrvd.cpp)
## geogram
pybind11_add_module(backend_geogram SHARED ${SOURCE_DIR}/backend_geogram.cpp)
target_link_libraries(backend_geogram PRIVATE exploragram)
## laguerre
pybind11_add_module(backend_laguerre SHARED ${SOURCE_DIR}/backend_laguerre.cpp)

## aabb_tree
pybind11_add_module(aabb_tree SHARED ${SOURCE_DIR}/aabb_tree.cpp)

## cgal_utils
pybind11_add_module(cgal_utils SHARED ${SOURCE_DIR}/cgal_utils.cpp)
