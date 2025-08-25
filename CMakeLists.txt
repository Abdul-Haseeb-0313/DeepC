cmake_minimum_required(VERSION 3.10)

# Project name
project(DeepC VERSION 0.1 LANGUAGES C)

# Add include directory
include_directories(${CMAKE_SOURCE_DIR}/includes)

# Collect all source files in src/
file(GLOB SOURCES src/*.c)

# Build static library (libdeepC.a)
add_library(deepC STATIC ${SOURCES})

# Install the library into system directories
install(TARGETS deepC
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

# Install headers into system include directory
install(DIRECTORY includes/ DESTINATION include)
