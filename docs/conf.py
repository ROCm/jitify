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

from rocm_docs import ROCmDocs

project = 'jitify'
copyright = 'Copyright (c) 2023 Advanced Micro Devices, Inc.'
os_support = ["linux"]
author = 'Jitify authors'
date = "2023-09-05"

external_projects_remote_repository = ""
external_projects_current_project = project

article_pages = [
        {
            "file" : "index",
            "os" : os_support,
            "author" : author,
            "date" : date,
            "read-time" : "1 min read",
        },
        {
            "file" : "user_guide/0_basic_usage",
            "os" : os_support,
            "author" : author,
            "date" : date,
            "read-time" : "1 min read",
        },
        {
            "file" : "user_guide/1_error_handling",
            "os" : os_support,
            "author" : author,
            "date" : date,
            "read-time" : "1 min read",
        },
        {
            "file" : "user_guide/2_basic_workflow",
            "os" : os_support,
            "author" : author,
            "date" : date,
            "read-time" : "1 min read",
        },
        {
            "file" : "user_guide/3_advanced_workflow",
            "os" : os_support,
            "author" : author,
            "date" : date,
            "read-time" : "1 min read",
        },
        {
            "file" : "user_guide/4_unit_tests",
            "os" : os_support,
            "author" : author,
            "date" : date,
            "read-time" : "1 min read",
        },
        {
            "file" : "user_guide/5_build_options",
            "os" : os_support,
            "author" : author,
            "date" : date,
            "read-time" : "1 min read",
        },
        {

            "file" : "user_guide/6_compiler_options",
            "os" : os_support,
            "author" : author,
            "date" : date,
            "read-time" : "2 min read",
        },
]

docs_core = ROCmDocs(project)
docs_core.run_doxygen(doxygen_root="./doxygen/", doxygen_path="doxygen/xml")  # Only if Doxygen is required for this project
docs_core.enable_api_reference()
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
