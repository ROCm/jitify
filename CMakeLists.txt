# Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.9)
project(jitify LANGUAGES CXX HIP)

# Todo(HIP): Eventually use HIP_INCLUDE_DIR instead of LIBHIPCXX_DIR once libhipcxx is part of rocm
# Todo(HIP): Integrate into rapids-cmake
# Set the path to libhipcxx GitHub repository inside the build directory
set(LIBHIPCXX_DIR ${CMAKE_BINARY_DIR}/libhipcxx)

# Download libhipcxx repository using Git
find_package(Git REQUIRED)
execute_process(
	COMMAND ${GIT_EXECUTABLE} clone --branch main https://github.com/ROCm/libhipcxx.git ${LIBHIPCXX_DIR}
)

# C++ compiler options.
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-unused-variable -Wno-sign-conversion") # Todo(HIP): enable warnings with -Wconversion later; clang is significantly more pendantic compared to other host compilers (say GCC)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

find_package(hip REQUIRED)
message(STATUS "HIP: HIP_INCLUDE_DIR = '${HIP_INCLUDE_DIR}'")

# Add macro definitions used in tests.
add_compile_definitions(
    HIP_INC_DIR="${HIP_INCLUDE_DIR}"
    HIPCUB_DIR="${HIP_INCLUDE_DIR}/hipcub")

# Copy the example_headers directory for use at runtime by tests.
file(COPY example_headers DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# -----------
# Google Test
# -----------
# Download and unpack googletest at configure time.
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()
# Prevent overriding the parent project's compiler/linker
# settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# ----
# Executable utilities
# ----
add_executable(stringify stringify.cpp)
function(add_stringify_command arg)
  add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${arg}.jit
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND ./stringify ${CMAKE_CURRENT_SOURCE_DIR}/${arg} > ${arg}.jit
      DEPENDS stringify)
endfunction()
add_executable(jitify2_preprocess jitify2_preprocess.cpp)
target_include_directories(jitify2_preprocess PRIVATE ${HIP_INCLUDE_DIRS})
if (NOT WIN32)
  find_package(Threads REQUIRED)
  target_link_libraries(jitify2_preprocess PRIVATE ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(jitify2_preprocess PRIVATE hip::host)
endif()

# ----
# Generate files for use in tests
# ----
add_stringify_command(example_headers/my_header1.cuh)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/jitify2_test_kernels.cu.jit.hpp
           ${CMAKE_CURRENT_BINARY_DIR}/jitify2_test_kernels.cu.headers.jit.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/jitify2_preprocess -i --minify
                -o ${CMAKE_CURRENT_BINARY_DIR}
                -s jitify2_test_kernels.cu.headers
                jitify2_test_kernels.cu
    DEPENDS jitify2_preprocess)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/jitify_2nd_compilation_unit.cu
     "#include \"jitify2.hpp\"")

if(NOT WIN32)
  set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
  set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
endif()

# -----
# Tests
# -----
enable_testing()
set(TESTS
    jitify2_test
)
foreach(test ${TESTS})
  # Note that generated headers are listed as source files to force dependency.
  add_executable(
      ${test} EXCLUDE_FROM_ALL ${test}.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/jitify_2nd_compilation_unit.cu
      ${CMAKE_CURRENT_BINARY_DIR}/example_headers/my_header1.cuh.jit
      ${CMAKE_CURRENT_BINARY_DIR}/jitify2_test_kernels.cu.jit.hpp
      ${CMAKE_CURRENT_BINARY_DIR}/jitify2_test_kernels.cu.headers.jit.cpp)
  # Set LIBHIPCXX_INC_DIR env var to the path of the libhipcxx include directory
  target_compile_definitions(${test} PRIVATE LIBHIPCXX_INC_DIR="${LIBHIPCXX_DIR}/include/")
  target_compile_definitions(${test} PRIVATE HIP_INC_DIR="${HIP_INCLUDE_DIR}")
  message(STATUS "HIP: HIP_INCLUDE_DIR = '${HIP_INCLUDE_DIR}'")
  # Ensure the generated headers can be found.
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  # Ensure the main jitify header can be found.
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${test} gtest_main)
  set_property(TARGET ${test} PROPERTY HIP_ARCHITECTURES OFF)
  if (NOT WIN32)
    target_link_libraries(${test} ${CMAKE_DL_LIBS})
  endif()
  add_test(NAME ${test} COMMAND ${test})
endforeach(test)
# Add "check" command that *builds and* runs tests, with verbose output.
# (The default "test" command neither builds nor gives verbose output).
add_custom_target(check ALL COMMAND ${CMAKE_CTEST_COMMAND} --verbose
                  DEPENDS ${TESTS})

# ----
# Docs
# ----
find_package(Doxygen)
if(DOXYGEN_FOUND)
  # Copy source file so that Doxygen can find it.
  # TODO: A better approach would be to configure a Doxyfile.in template file
  # with the appropriate input name and/or output path.
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/jitify2.hpp
                 ${CMAKE_CURRENT_BINARY_DIR}/
                 COPYONLY)
  set(doxyfile ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile2)
  add_custom_target(
      doc
      COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${doxyfile})
endif()
