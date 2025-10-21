cmake_minimum_required(VERSION 3.19)
project(mymodule CXX)

# Module name
set(MODULE_NAME mymodule)
set(MODULE_VERSION_MAJOR 0)
set(MODULE_VERSION_MINOR 1)
set(MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}")

# Must provide the path to nest-config
if(NOT with-nest)
    message(FATAL_ERROR "-Dwith-nest=<nest_config> is required")
endif()
set(NEST_CONFIG ${with-nest})

# Get NEST variables
execute_process(COMMAND ${NEST_CONFIG} --compiler OUTPUT_VARIABLE NEST_COMPILER OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${NEST_CONFIG} --cflags OUTPUT_VARIABLE NEST_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${NEST_CONFIG} --includes OUTPUT_VARIABLE NEST_INCLUDES OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${NEST_CONFIG} --libs OUTPUT_VARIABLE NEST_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${NEST_CONFIG} --prefix OUTPUT_VARIABLE NEST_INSTALL_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${NEST_CONFIG} --libdir OUTPUT_VARIABLE NEST_LIBDIR OUTPUT_STRIP_TRAILING_WHITESPACE)

# Convert include flags into directories
if(NEST_INCLUDES)
    string(REPLACE " " ";" NEST_INCLUDES_LIST "${NEST_INCLUDES}")
    foreach(inc ${NEST_INCLUDES_LIST})
        if(inc MATCHES "^-I(.*)")
            string(REGEX REPLACE "^-I(.*)" "\\1" dir "${inc}")
            if(IS_DIRECTORY "${dir}")
                include_directories("${dir}")
            endif()
        endif()
    endforeach()
endif()

# Set compiler
set(CMAKE_CXX_COMPILER "${NEST_COMPILER}")

# Add src subdirectory
add_subdirectory(src)

# Installation variables
set(CMAKE_INSTALL_PREFIX "${NEST_INSTALL_PREFIX}" CACHE STRING "" FORCE)
set(CMAKE_INSTALL_LIBDIR "${NEST_LIBDIR}" CACHE STRING "" FORCE)
