#!/usr/bin/env bash
# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Run the jitify test suite against a tree produced by ci/internal/build.sh.
# The tester images reach this script through the source tree shipped inside
# the tests tarball.
#
# Required environment variables:
#   BUILD_DIR  Scratch directory the build tree was created in.
#
# Optional environment variables:
#   PROJECT_ID      Checkout directory name (default: jitify).
#   JITIFY_OPTIONS  Extra options handed to the JIT compilation tests, e.g.
#                   '--std=c++17' on manylinux.

set -xeu

: "${BUILD_DIR:?BUILD_DIR must be set}"

PROJECT_ID="${PROJECT_ID:-jitify}"
build_build_dir="${BUILD_DIR}/${PROJECT_ID}/build"

if [ -d "/opt/rh/gcc-toolset-$(g++ -dumpversion)/root/usr" ]; then
  toolchain="--gcc-toolchain=/opt/rh/gcc-toolset-$(g++ -dumpversion)/root/usr"
  export CCC_OVERRIDE_OPTIONS="+${toolchain}"
fi

cmake --build "${build_build_dir}" --target check
