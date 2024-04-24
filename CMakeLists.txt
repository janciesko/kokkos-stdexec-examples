# Kokkos Stdexec Examples (KSEE)

cmake_minimum_required(VERSION 3.12)
project(KokkosStdExecExamples VERSION 0.0.2)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)
include(${CMAKE_CURRENT_SOURCE_DIR}/modules/flags.cmake)

option(ENABLE_BUILD_CGSOLVE "Build cgsolve example" ON)
option(ENABLE_BUILD_FIBONACCI "Build fibonacci example" ON)
option(ENABLE_BUILD_SIMPLEGRAPH "Build simplegraph example" ON)
option(ENABLE_BUILD_SIMPLETASK "Build simpletasks example" ON)
option(ENABLE_BUILD_ULTBACKENDS "Build ultbackends example" ON)

option(ENABLE_QTHREADS "Build ultbackends example with Qthreads" OFF)
option(ENABLE_PICA "Build ultbackends example with PICA" OFF)
option(ENABLE_ARGOBOTS "Build ultbackends example with Argotobs" OFF)

set(KSEE_ENABLE_BUILD_CGSOLVE ${ENABLE_BUILD_CGSOLVE} CACHE BOOL "" FORCE)
set(KSEE_ENABLE_BUILD_FIBONACCI ${ENABLE_BUILD_FIBONACCI} CACHE BOOL "" FORCE)
set(KSEE_ENABLE_BUILD_SIMPLEGRAPH ${ENABLE_BUILD_SIMPLEGRAPH} CACHE BOOL "" FORCE)
set(KSEE_ENABLE_BUILD_SIMPLETASK ${ENABLE_BUILD_SIMPLETASK} CACHE BOOL "" FORCE)
set(KSEE_ENABLE_BUILD_ULTBACKENDS ${ENABLE_BUILD_ULTBACKENDS} CACHE BOOL "" FORCE)

set(KSEE_ENABLE_QTHREADS ${ENABLE_QTHREADS} CACHE BOOL "" FORCE)
set(KSEE_ENABLE_ENABLE_PIKA ${ENABLE_PIKA} CACHE BOOL "" FORCE)
set(KSEE_ENABLE_ENABLE_ARGOBOTS ${ENABLE_ARGOBOTS} CACHE BOOL "" FORCE)

find_package(stdexec REQUIRED)

if(KSEE_ENABLE_BUILD_CGSOLVE)
    message(STATUS "Including cgsolve")
    add_subdirectory(cgsolve)
endif()

if(KSEE_ENABLE_BUILD_FIBONACCI)
    message(STATUS "Including fibonacci")
    add_subdirectory(fibonacci)
endif()

if(KSEE_ENABLE_BUILD_SIMPLEGRAPH)
    message(STATUS "Including simplegraph")
    add_subdirectory(simplegraph)
endif()

if(KSEE_ENABLE_BUILD_SIMPLETASK)
    message(STATUS "Including simpletask")
    add_subdirectory(simpletask)
endif()

if(KSEE_ENABLE_BUILD_ULTBACKENDS)
    message(STATUS "Including ultbackends")
    add_subdirectory(ultbackends)
endif()

