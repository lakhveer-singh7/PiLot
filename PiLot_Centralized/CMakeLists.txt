cmake_minimum_required(VERSION 3.20)
project(PiLot_Centralized C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type defaults
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_DEBUG   "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_C_FLAGS_RELEASE "-O2 -Wall -Wextra -DNDEBUG")

# Header search path
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.c
    src/logging.c
    src/augmentation.c
    src/data/tensor.c
    src/data/ucr_loader.c
    src/config/config_loader.c
    src/nn/conv1d.c
    src/nn/pooling.c
    src/nn/fully_connected.c
    src/nn/activations.c
    src/nn/optimizers.c
)

add_executable(pilot_centralized ${SOURCES})

# Link math library (and pthread for clock_gettime on some systems)
target_link_libraries(pilot_centralized m pthread)
