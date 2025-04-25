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

<a name="basic_workflow"/>

# Basic workflow example

Here we describe a complete workflow for integrating Jitify into an
application. There are many ways to use Jitify, but this is the
recommended approach.

The jitify_preprocess tool allows HIP source to be transformed and
headers to be loaded and baked into the application during offline
compilation, avoiding the need to perform these transformations or
to load any headers at runtime.

First run jitify_preprocess to generate JIT headers for your runtime
sources:

```bash
$ ./jitify_preprocess -i myprog1.cu myprog2.cu
```

Then include the headers in your application:

```c++
#include "myprog1.cu.jit.hpp"
#include "myprog2.cu.jit.hpp"
```

And use the variables they define to construct a `ProgramCache` object:

```c++
  using jitify2::ProgramCache;
  static ProgramCache<> myprog1_cache(/*max_size = */ 100, *myprog1_cu_jit);
```

Kernels can then be obtained directly from the cache:

```c++
  using jitify2::reflection::Template;
  using jitify2::reflection::Type;
  myprog1_cache
    .get_kernel(Template("my_kernel").instantiate(123, Type<float>()))
    ->configure(grid, block)
    ->launch(idata, odata);
```
