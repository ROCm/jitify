#!/bin/bash

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

##############################################i###
# jitify GPU build and test script for CI #
##################################################
set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}


# Set working dir to jitify root
export JITIFY_ROOT="$(cd "$(dirname "$0")" && cd ../ && pwd)"

################################################################################
# SETUP - Check environment
################################################################################

echo "Check environment"
env

echo "Check that Github access token is set up"
if [ -z ${GITHUB_PASS} ]
then 
  echo "Error: You need to set GITHUB_PASS (personal access token) to install jitify dependencies via rapids-cmake from github"	
  exit -1
fi

if [ -z ${GITHUB_USER} ]
then 
  echo "Error: You need to set GITHUB_USER (personal access token) to install jitify dependencies via rapids-cmake from github"	
  exit -1
fi

echo "Check GPU usage"
rocm-smi

echo "Check versions"
cmake --version

################################################################################
# BUILD - Build from Source
################################################################################

echo "Build and Run Tests"
cd ${JITIFY_ROOT}
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=hipcc ..
make

