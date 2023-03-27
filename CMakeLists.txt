cmake_minimum_required(VERSION 3.20)

project(ccta)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the path to the PyTorch C++ libraries
# Alternatively set the -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch/share/cmake
# set(Torch_DIR "/Users/josqealonsosolis-lemus/opt/anaconda3/envs/ccta_env/lib/python3.9/site-packages/torch/share/cmake")

# Find the PyTorch C++ libraries
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if (NOT TORCH_FOUND) 
    message(STATUS "This project Requires ITK and will not be compiled")
    return()
endif()

# find_package(ITK)

# if (NOT ITK_FOUND)
#     message(STATUS "This project Requires ITK and will not be compiled")
#     return()
# endif()

# Add the subdirectories
set(CCTA_INCLUDES "./ccta_lib/include")

add_subdirectory(ccta_lib)
include_directories(${CCTA_INCLUDES})
set(SOURCES ${CCTA_SRCS})

# Set the output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(ccta main.cpp)
# ${ITK_INCLUDE_DIRS})
target_include_directories(ccta_lib PUBLIC ${TORCH_INCLUDE_DIRS} )
target_include_directories(ccta PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(ccta ccta_lib "${TORCH_LIBRARIES}" )
set_property(TARGET ccta PROPERTY CXX_STANDARD 17) 