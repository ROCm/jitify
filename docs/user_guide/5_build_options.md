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

<a name="build_options"/>

# Build options

- `JITIFY_ENABLE_EXCEPTIONS=1`

  Defining this macro to 0 before including the jitify header disables
  the use of exceptions throughout the API, requiring the user to
  explicitly check for errors. See [Error handling](#error_handling)
  for more details.

- `JITIFY_THREAD_SAFE=1`

  Defining this macro to 0 before including the jitify header disables
  the use of mutexes in the ProgramCache class.

- `JITIFY_LINK_HIPRTC_STATIC=0`

  Defining this macro to 1 before including the jitify header disables
  dynamic loading of the HIPRTC dynamic library and allows the
  library to be linked statically.

- `JITIFY_LINK_HIP_STATIC=0`

  Defining this macro to 1 before including the jitify header disables
  dynamic loading of the HIP dynamic library and allows the
  library to be linked statically.

- `JITIFY_FAIL_IMMEDIATELY=0`

  Defining this macro to 1 before including the jitify header causes
  errors to trigger exceptions/termination immediately instead of
  only when a jitify object is dereferenced. This is useful for
  debugging, as it allows the origin of an error to be found via a
  backtrace.

