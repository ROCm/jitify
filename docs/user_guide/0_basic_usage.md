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

<a name="basic_usage"/>

# Basic usage

Jitify is just a single header file:

```c++
#include <jitify2.hpp>
```

It does not have any link-time dependencies besides the dynamic loader (which is used to load the HIP and HIPRTC libraries at runtime), so compilation is simple:

```bash
# with hipcc:
$ hipcc ... -ldl
```

It provides a simple API for compiling and executing HIP source code at runtime:

```c++
  std::string program_name = "my_program";
  std::string program_source = R"(
  template <typename T>
  __global__ void my_kernel(T* data) { *data = T{7}; }
  )";
  dim3 grid(1), block(1);
  float* data;
  hipMalloc((void**)&data, sizeof(float));
  auto program =
      jitify2::Program(program_name, program_source)
          // Preprocess source code and load all included headers.
          ->preprocess({"-std=c++14"})
          // Compile, link, and load the program, and obtain the loaded kernel.
          ->get_kernel("my_kernel<float>")
          // Configure the kernel launch.
          ->configure(grid, block)
          // Launch the kernel.
          ->launch(data);
```
