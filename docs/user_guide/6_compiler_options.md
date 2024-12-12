<!---
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
-->

<a name="compiler_options"/>

# Compiler options

The Jitify API accepts options that can be used to control compilation
and linking. While most options are simply passed through to HIPRTC
(for compiler and linker options), some trigger special behavior in
Jitify as detailed below:

- `-I<dir>`

  Specifies a directory to search for include files. Jitify intercepts
  these flags and handles searching for include files itself instead
  of relying on HIPRTC, in order to provide more flexibi

- `--offload-arch=gfx*`or `--gpu-architecture=gfx*` or `--gpu-name=gfx*` 

  Specifies the target AMD GPU architecture for code object generation
  during JIT-compilation. If ommitted, the target architecture will be
  queried from the current HIP device, as provided by hipCtxGetDevice().

- `-std=<std>`

  Unless otherwise specified, this flag is automatically passed to
  HIPRTC for all kernels and is set to `c++11` (which is the minimum
  requirement for Jitify itself). Jitify also supports the value
  `-std=c++03` for explicitly selecting the `C++03` standard.

- `--minify (-m)`

  This option is supported by Jitify only at preprocessing time and
  causes all runtime source code and headers to be "minified"
  (all comments and most whitespace is removed). This reduces the
  size of the source code and achieves a basic level of code
  obfuscation.

- `--no-replace-pragma-once (-no-replace-pragma-once)`

  This option is supported by Jitify only at preprocessing time and
  disables the automatic replacement of `#pragma once` with
  `#ifndef ...`.

- `--no-system-headers-workaround (-no-system-headers-workaround)`
  
  In the HIP compilation path, HIPRTC might search for system headers
  in /usr/include. For headers like <iterator> this can yield compiler errors
  due to duplicate symbol definitions (usually conflicts with hip/amd_detail/amd_hip_vector_types.h).
  Per default, jitify therefore uses jitsafe header implementations that work around these issues.
  This workaround can be disabled with this option.

- `--hip-std (-hip-std)`

  [EXPERIMENTAL]
  This option is supported by Jitify only at preprocessing time and
  causes all instances of `std::foo` to be automatically replaced
  with `::hip::std::foo`, with the intention of supporting the use of
  the libhipcxx header implementations instead of the system
  implementations. This is experimental because it does not currently
  support the transformation of `namespace std {` (as is used for
  specializations of standard library templates).

Linker options:

- `-l<library>`

  Specifies a device library to link with. This can be a static
  library, in which case the prefix/suffix will be added automatically
  if none is present (e.g., `-lfoo` is equivalent to `-llibfoo.a` in
  Linux systems), or a .bc/.fatbin/.o file, in which case the
  file type will be inferred from the extension.

- `-L<dir>`

  Specifies a directory to search for linker files. For a given
  `-l<library>`, the unmodified `<library>` name is tried first,
  before searching for the file in the `-L` directories in the order
  they are listed.

The following linker options are mapped directly to options in the
hiprtc APIs.

- `-g`

  Enables the generation of debug information.

- `--opt-level=N (-O)`

  Specifies the optimization level.

- `--verbose (-v)`

  Enables verbose logging.

