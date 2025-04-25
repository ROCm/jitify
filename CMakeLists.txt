# Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

option(JITIFY_ONLY_CREATE_PACKAGE "Package the `jitify` subfolder." OFF)
option(JITIFY_ENABLE_LIBHIPCXX_TESTS "Test the C++ header JIT compilation." ON)

if(JITIFY_ENABLE_LIBHIPCXX_TESTS OR JITIFY_ONLY_CREATE_PACKAGE)
  set(
    RAPIDS_CMAKE_MODULE_PATH
    $ENV{RAPIDS_CMAKE_MODULE_PATH}
    CACHE FILEPATH
    "Announce that ROCmDS-CMake is available via the provided module path."
  )
  if (NOT "${RAPIDS_CMAKE_MODULE_PATH}" STREQUAL "")
    list(APPEND CMAKE_MODULE_PATH "${RAPIDS_CMAKE_MODULE_PATH}")
  else()
    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/JITIFY_RAPIDS.cmake)
      if(DEFINED ENV{RAPIDS_CMAKE_SCRIPT_REPO})
        set(RAPIDS_CMAKE_SCRIPT_REPO "$ENV{RAPIDS_CMAKE_SCRIPT_REPO}")
      else()
        set(RAPIDS_CMAKE_SCRIPT_REPO ROCm-DS/ROCmDS-CMake)
      endif()
      if(DEFINED ENV{RAPIDS_CMAKE_SCRIPT_BRANCH})
        set(RAPIDS_CMAKE_SCRIPT_BRANCH "$ENV{RAPIDS_CMAKE_SCRIPT_BRANCH}")
      else()
        set(RAPIDS_CMAKE_SCRIPT_BRANCH release/1.0.x)
      endif()

      set(URL "https://raw.githubusercontent.com/${RAPIDS_CMAKE_SCRIPT_REPO}/${RAPIDS_CMAKE_SCRIPT_BRANCH}/RAPIDS.cmake")
      file(DOWNLOAD ${URL}
        ${CMAKE_CURRENT_BINARY_DIR}/JITIFY_RAPIDS.cmake
        STATUS DOWNLOAD_STATUS
      )
      list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
      list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)

      if(${STATUS_CODE} EQUAL 0)
        message(STATUS "Downloaded 'JITIFY_RAPIDS.cmake' successfully!")
      else()
        file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/JITIFY_RAPIDS.cmake)
        # for debuging: message(FATAL_ERROR "Failed to download 'JITIFY_RAPIDS.cmake'. URL: ${URL}, Reason: ${ERROR_MESSAGE}")
        message(FATAL_ERROR "Failed to download 'JITIFY_RAPIDS.cmake'. Reason: ${ERROR_MESSAGE}")
      endif()
    endif()

    if(DEFINED ENV{RAPIDS_CMAKE_BRANCH})
      set(rapids-cmake-branch $ENV{RAPIDS_CMAKE_BRANCH})
    endif()

    include(${CMAKE_CURRENT_BINARY_DIR}/JITIFY_RAPIDS.cmake)
  endif()

  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-export)

  rapids_cpm_init()
endif()

project(jitify VERSION 0.0 LANGUAGES CXX HIP)


if(JITIFY_ENABLE_LIBHIPCXX_TESTS)
  # Obtain libhipcxx via rapids-cmake

  # prescribe dependencies
  include(cpm/package_override)
  rapids_cpm_package_override(${CMAKE_CURRENT_LIST_DIR}/versions.json)

  # download/find libhipcxx CMake package
  # Todo(HIP): Eventually use HIP_INCLUDE_DIR instead of LIBHIPCXX_DIR once libhipcxx is part of rocm
  include(${rapids-cmake-dir}/cpm/libhipcxx.cmake)

  # NOTE(HIP/AMD): We do not include libhipcxx headers
  # into the export sets when packaging as
  # the libhipcxx headers are only needed for the unit tests.
  rapids_cpm_libhipcxx()
  if(DEFINED _libhipcxx_INCLUDE_DIR)
    # when using local package from conda env
    set(LIBHIPCXX_INC_DIR ${_libhipcxx_INCLUDE_DIR})
  else()
    set(LIBHIPCXX_INC_DIR "${libhipcxx_SOURCE_DIR}/include/")
  endif()
endif()

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
  if(JITIFY_ENABLE_LIBHIPCXX_TESTS)
    target_compile_definitions(${test} PRIVATE LIBHIPCXX_INC_DIR="${LIBHIPCXX_INC_DIR}")
    target_compile_definitions(${test} PRIVATE JITIFY_ENABLE_LIBHIPCXX_TESTS)
  endif()
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
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose
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

#------
# Packaging
# NOTE We need to consider to move to ROCm CMake from raw CPACK
if(JITIFY_ONLY_CREATE_PACKAGE)
  add_library(jitify INTERFACE)
  install(TARGETS jitify EXPORT jitify-targets)

  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/jitify2.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/jitify2_hip.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/jitify2_cuda.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/jitify2_preprocess.cpp" DESTINATION include)

  rapids_export(
        INSTALL jitify
        EXPORT_SET jitify-targets
        GLOBAL_TARGETS jitify
        NAMESPACE jitify::
        )

  set(DEFAULT_JITIFY_DEVEL_PACKAGE_SUFFIX "dev")
  if (DEFINED ENV{JITIFY_DEVEL_PACKAGE_SUFFIX})
    set(DEFAULT_JITIFY_DEVEL_PACKAGE_SUFFIX "$ENV{JITIFY_DEVEL_PACKAGE_SUFFIX}")
  endif()

  set(JITIFY_DEVEL_PACKAGE_SUFFIX "${DEFAULT_JITIFY_DEVEL_PACKAGE_SUFFIX}" CACHE STRING "Overwrite development package suffix.")
  set(CPACK_PACKAGE_NAME "rocm-hip-jitify-${JITIFY_DEVEL_PACKAGE_SUFFIX}" CACHE STRING "Overwrite package name.")
  set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_VENDOR}")
  set(CPACK_OUTPUT_FILE_PREFIX "dist" CACHE PATH "Package output file prefix.")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.TXT")
  set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
  set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
  set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm-ds" CACHE PATH "Package install prefix.")
  include(CPack)
endif()
