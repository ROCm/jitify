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

<a name="advanced_workflow"/>

# Advanced workflow example

The jitify_preprocess tool also supports automatic minification of source code as
well as generation of a separate source file for sharing runtime headers between
different runtime programs:

```bash
$ ./jitify_preprocess -i --minify -s myheaders myprog1.cu myprog2.cu
```

The generated source file should be linked with your application:

```bash
$ g++ -o myapp myapp.cpp myheaders.jit.cpp ...
```

And the generated variable should be passed to the ProgramCache constructor.
A directory name can also be specified to enable caching of compiled binaries on
disk:

```c++
#include "myprog1.cu.jit.hpp"
#include "myprog2.cu.jit.hpp"
...
  using jitify2::ProgramCache;
  static ProgramCache<> myprog1_cache(
      /*max_size = */ 100, *myprog1_cu_jit, myheaders_jit, "/tmp/my_jit_cache");
```

For advanced use-cases, multiple kernels can be instantiated in a single program:

```c++
  using jitify2::reflection::Template;
  using jitify2::reflection::Type;
  using jitify2::Program;
  std::string kernel1 = Template("my_kernel1").instantiate(123, Type<float>());
  std::string kernel2 =
      Template("my_kernel2").instantiate(45, Type<int>(), Type<int>());
  Program myprog1 = myprog1_cache.get_program({kernel1, kernel2});
  myprog1->set_global_value("my::value", 3.14f);
  myprog1->get_kernel(kernel1)->configure(grid, block)->launch(idata, odata);
  myprog1->get_kernel(kernel2)->configure(grid, block)->launch(idata, odata);
```

For improved performance, the cache can be given user-defined keys:

```c++
  using jitify2::ProgramCache;
  using jitify2::Kernel;
  using MyKeyType = uint32_t;
  static ProgramCache<MyKeyType> myprog1_cache(
      /*max_size = */ 100, *myprog1_cu_jit, myheaders_jit, "/tmp/my_jit_cache");
  std::string kernel1 = Template("my_kernel1").instantiate(123, Type<float>());
  Kernel kernel = myprog1_cache.get_kernel(MyKeyType(7), kernel1);
```
