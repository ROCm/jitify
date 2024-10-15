<!---
 Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
-->
# jitify
A single-header C++ library for simplifying the use of HIP Runtime Compilation (HIPRTC). 
This is a port of the original CUDA version at https://github.com/NVIDIA/jitify/tree/jitify2 to HIP in order to enable support for AMD GPUs.

# Requirements
- ROCm and HIP 6.2.0 or higher
- CMake 3.9 or higher (for building the tests)
- AMD MI100/MI200 GPUs
- Linux distribution (tested with Ubuntu 20.04+)

# Current Limitations
- No support for Windows.
- JIT compilation of HIP sources for NVIDIA architectures is presently not supported.
- LTO is not supported.
- Linking JIT-compiled code to code in the current executable is not supported.
- Removing unused globals is presently not supported.
- Some features are only partially supported due to several issues we have found with HIPRTC. Please see [here](#unittests) for more details.

<a name="unittests"></a>
# Status of unit tests

The following tests are only supported partially or have been disabled or they have been modified:

| Test | Status | Comment |  Related to Ticket |
| ---- | --------- | ------- | -------------------- |
| Minify | modified | Header includes in jit-source were disabled, as they yield errors during JIT-compilation |  SWDEV-419480  |
| AssertHeader | disabled | Failing assert crashes the host process with HIP, so the test would fail with HIP. This is a different behaviour between CUDA and HIP. | n/a |
| ConstantMemory | modified | Currently, we do not have a way to extract mangled symbol names from some intermediate code representation (like ptx on NVIDIA side). Symbols therefore are kept tracked of by adding their name expressions manually in the test. | n/a |
| LinkCurrentExecutable | disabled | Hiprtc presently does not allow to link input type HIPRTC_JIT_INPUT_OBJECT. | SWDEV-419737 |
| RemovedUnusedGlobals | disabled | Checks CUDA-/NVRTC-specific features: Currently, we do not have a way to extract unused global variables from some intermediate code representation (like ptx on NVIDIA side). |  |
| ArchFlags | modified | We cannot get arch name on the device with __HIP_ARCH__.  |  |
| CuRandKernel | disabled | Currently hiprtc returns redefinition errors when including thrust/hipcub/hiprand_kernel headers | SWDEV-419480 |
| Thrust | disabled | Currently hiprtc returns redefinition errors when including thrust/hipcub/hiprand_kernel headers | SWDEV-419480 |
| CompileLTO_IR | disabled | This test is not supported as LTO is not available in HIPRTC. | n/a |
| LinkLTO | disabled | This test is not supported as LTO is not available in HIPRTC. | n/a |


