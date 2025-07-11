
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

<a name="error_handling"/>

# Error handling

All Jitify APIs such as `preprocess()`, `compile()`, `link()`,
`load()`, and `get_kernel()` return special objects that wrap either a
valid data object (if the call succeeds) or an error state (if the
call fails). The error state can be inspected using `operator bool()`
and the `error()` method. If the macro `JITIFY_ENABLE_EXCEPTIONS` is not
defined to 0 before jitify.hpp is included in your application, an
exception will be thrown when attempting to use the result of a failed
call or when a method such as `launch()` fails:

```c++
  jitify2::PreprocessedProgram preprog =
      jitify2::Program(program_name, program_source)
          ->preprocess({"-std=c++14"});
  if (!preprog) {
    // The call failed, we can access the error.
    std::cerr << preprog.error() << std::endl;
    // This will either throw an exception or terminate the application.
    *preprog;
  } else {
    // The call succeeded, we can access the data object.
    jitify2::PreprocessedProgramData preprog_data = *preprog;
    // Or we can directly call a method on the data object.
    jitify2::CompiledProgram compiled = preprog->compile("my_kernel<float>");
    // This will throw (or terminate) if any of the chained methods fails.
    preprog->compile("my_kernel<float>")
        ->link()
        ->load()
        ->get_kernel("my_kernel<float>")
        ->configure(1, 1)
        ->launch();
  }
```

