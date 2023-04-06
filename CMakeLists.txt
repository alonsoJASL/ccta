cmake_minimum_required(VERSION 3.20)
project(ccta)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the path to the PyTorch C++ libraries
# Alternatively set the -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch/share/cmake
set(Torch_DIR "$HOME/dev/libraries/libtorch-arm64/share/cmake/Torch")

# Find the PyTorch C++ libraries
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if (NOT TORCH_FOUND) 
    message(STATUS "This project Requires ITK and will not be compiled")
    return()
endif()

find_package(ITK)
include(${ITK_USE_FILE})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ITK_CXX_FLAGS}")

# Add the subdirectories
set(CCTA_INCLUDES "./ccta++C/include")

add_subdirectory(ccta++/src)
include_directories(${CCTA_INCLUDES})
set(SOURCES ${CCTA_SRCS})

# Set the output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(ccta ccta++ ${TORCH_LIBRARIES} ${ITK_LIBRARIES})
set_property(TARGET ccta PROPERTY CXX_STANDARD 17)
