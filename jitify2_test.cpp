/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "hip/hip_runtime.h"

#define JITIFY_ENABLE_EXCEPTIONS 1
#include "jitify2.hpp"

#include "example_headers/class_arg_kernel.cuh"
#include "example_headers/my_header1.cuh.jit"
#include "jitify2_test_kernels.cu.jit.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#define CHECK_HIP(call)                                                   \
  do {                                                                    \
    hipError_t status = call;                                             \
    if (status != hipSuccess) {                                           \
      const char* str;                                                    \
      hip().GetErrorName()(status, &str);                                 \
      std::cout << "(HIP) returned " << str;                              \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      ASSERT_EQ(status, hipSuccess);                                      \
    }                                                                     \
  } while (0)

#define CHECK_HIPRT(call)                                                 \
  do {                                                                    \
    hipError_t status = call;                                             \
    if (status != hipSuccess) {                                           \
      std::cout << "(HIPRT) returned " << hipGetErrorString(status);      \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      ASSERT_EQ(status, hipSuccess);                                      \
    }                                                                     \
  } while (0)

using namespace jitify2;
using namespace jitify2::reflection;

template <typename ValueType, typename ErrorType>
std::string get_error(
    const jitify2::detail::FallibleValue<ValueType, ErrorType>& x) {
  if (x) return "";
  return x.error();
}

void debug_print(const StringVec& v, const std::string& varname) {
  std::cerr << "--- BEGIN VECTOR " << varname << " ---\n";
  for (const auto& x : v) {
    std::cerr << x << "\n";
  }
  std::cerr << "--- END VECTOR " << varname << " ---" << std::endl;
}

bool contains(const StringVec& v, const std::string& s, const char* varname) {
  bool result = std::find(v.begin(), v.end(), s) != v.end();
  if (!result) debug_print(v, varname);
  return result;
}
bool not_contains(const StringVec& v, const std::string& s,
                  const char* varname) {
  bool result = std::find(v.begin(), v.end(), s) == v.end();
  if (!result) debug_print(v, varname);
  return result;
}

#define CONTAINS(src, target) contains(src, target, #src)
#define NOT_CONTAINS(src, target) not_contains(src, target, #src)

TEST(Jitify2Test, Simple) {
  static const char* const source = R"(
template <int N, typename T>
__global__ void my_kernel(T* data) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  T data0 = data[0];
  for( int i=0; i<N-1; ++i ) {
    data[0] *= data0;
  }
})";
  using dtype = float;
  dtype* d_data;
  CHECK_HIPRT(hipMalloc((void**)&d_data, sizeof(dtype)));
  // Test serialization.
  auto program =
      Program::deserialize(Program("my_program", source)->serialize());
  ASSERT_EQ(get_error(program), "");
  auto preprog =
      PreprocessedProgram::deserialize(program->preprocess()->serialize());
  ASSERT_EQ(get_error(preprog), "");
  std::string kernel_inst =
      Template("my_kernel").instantiate(3, type_of(*d_data));
  // todo(hip): WAR for SWDEV-379212
  kernel_inst = "my_kernel<3,float>";

  auto compiled =
      CompiledProgram::deserialize(preprog->compile(kernel_inst)->serialize());
  ASSERT_EQ(get_error(compiled), "");
  auto linked = LinkedProgram::deserialize(compiled->link()->serialize());
  ASSERT_EQ(get_error(linked), "");

  // Test that kernel instantiation produces correct result.
  Kernel kernel = linked->load()->get_kernel(kernel_inst);
  dim3 grid(1), block(1);
  dtype h_data = 5;
  CHECK_HIPRT(hipMemcpy(d_data, &h_data, sizeof(dtype), hipMemcpyHostToDevice));
  ASSERT_EQ(kernel->configure(grid, block)->launch(d_data), "");
  CHECK_HIPRT(hipMemcpy(&h_data, d_data, sizeof(dtype), hipMemcpyDeviceToHost));
  EXPECT_FLOAT_EQ(h_data, 125.f);

  h_data = 5;
  CHECK_HIPRT(hipMemcpy(d_data, &h_data, sizeof(dtype), hipMemcpyHostToDevice));
  ASSERT_EQ(kernel->configure_1d_max_occupancy()->launch(d_data), "");
  CHECK_HIPRT(hipMemcpy(&h_data, d_data, sizeof(dtype), hipMemcpyDeviceToHost));
  EXPECT_FLOAT_EQ(h_data, 125.f);

  CHECK_HIPRT(hipFree(d_data));
}

bool header_callback(const std::string& filename, std::string* source) {
  // On success, write to *source and return true, otherwise return false.
  if (filename == "example_headers/my_header4.cuh") {
    *source = R"(
#pragma once
template <typename T>
__device__ T pointless_func(T x) {
  return x;
};)";
    return true;
  } else {
    // Find this file through other mechanisms.
    return false;
  }
}

// Returns, e.g., "gfx90a" for an MI200 device
std::string get_current_device_arch() {
  hipDevice_t device = 0;
  hipDeviceProp_t device_prop;
  std::string arch_name;
  hipError_t ret;

  if ((ret = hipGetDeviceProperties(&device_prop, device)) != hipSuccess) {
    return std::to_string(ret);
  }
  arch_name = device_prop.gcnArchName;
  size_t startPos = 0;
  size_t endPos = arch_name.find(":", startPos);
  if (endPos == std::string::npos) {
    return "gfx not found in the arch_name string";
  }
  arch_name = arch_name.substr(startPos, endPos - (startPos));
  return arch_name;
}

TEST(Jitify2Test, MultipleKernels) {
  static const char* const source = R"(
#include "example_headers/my_header1.cuh"
#include "example_headers/my_header2.cuh"
#include "example_headers/my_header3.cuh"
#include "example_headers/my_header4.cuh"

__global__ void my_kernel1(const float* indata, float* outdata) {
  outdata[0] = indata[0] + 1;
  outdata[0] -= 1;
}

template <int C, typename T>
__global__ void my_kernel2(const float* indata, float* outdata) {
  for (int i = 0; i < C; ++i) {
    outdata[0] = pointless_func(identity(sqrt(square(negate(indata[0])))));
  }
})";

  enum { C = 123 };
  typedef float T;
  std::string kernel2_inst =
      Template("my_kernel2").instantiate<NonType<int, C>, T>();
  LoadedProgram program = Program("multiple_kernels_program", source)
                              ->preprocess({}, {}, header_callback)
                              ->load({"my_kernel1", kernel2_inst});
  ASSERT_EQ(get_error(program), "");

  T* indata;
  T* outdata;
  CHECK_HIPRT(hipMalloc((void**)&indata, sizeof(T)));
  CHECK_HIPRT(hipMalloc((void**)&outdata, sizeof(T)));
  T inval = 3.14159f;
  CHECK_HIPRT(hipMemcpy(indata, &inval, sizeof(T), hipMemcpyHostToDevice));

  dim3 grid(1), block(1);
  ASSERT_EQ(program->get_kernel("my_kernel1")
                ->configure(grid, block)
                ->launch(indata, outdata),
            "");
  // These invocations are all equivalent.
  ASSERT_EQ(program->get_kernel(kernel2_inst)
                ->configure(grid, block)
                ->launch(indata, outdata),
            "");
  ASSERT_EQ(program
                ->get_kernel(Template("my_kernel2")
                                 .instantiate({reflect((int)C), reflect<T>()}))
                ->configure(grid, block)
                ->launch(indata, outdata),
            "");
  ASSERT_EQ(
      program->get_kernel(Template("my_kernel2").instantiate((int)C, Type<T>()))
          ->configure(grid, block)
          ->launch(indata, outdata),
      "");
  ASSERT_EQ(
      program
          ->get_kernel(
              Template("my_kernel2").instantiate((int)C, type_of(*indata)))
          ->configure(grid, block)
          ->launch(indata, outdata),
      "");
  ASSERT_EQ(
      program
          ->get_kernel(
              Template("my_kernel2").instantiate((int)C, instance_of(*indata)))
          ->configure(grid, block)
          ->launch(indata, outdata),
      "");

  T outval = 0;
  CHECK_HIPRT(hipMemcpy(&outval, outdata, sizeof(T), hipMemcpyDeviceToHost));
  CHECK_HIPRT(hipFree(outdata));
  CHECK_HIPRT(hipFree(indata));

  EXPECT_FLOAT_EQ(inval, outval);
}

TEST(Jitify2Test, LaunchLatencyBenchmark) {
  static const char* const source = R"(
template <int N, int M, typename T, typename U>
__global__ void my_kernel(const T*, U*) {}
)";
  const size_t max_size = 2;
  // Note: It's faster (by ~300ns) to use custom keys, but we want to test
  // worst-case perf.
  ProgramCache<> cache(max_size, *Program("my_program", source)->preprocess(),
                       nullptr);
  float* idata = nullptr;
  uint8_t* odata = nullptr;
  dim3 grid(1), block(1);
  Kernel kernel = cache.get_kernel(
      Template("my_kernel")
          .instantiate(3, 4, type_of(*idata), type_of(*odata)));
  ASSERT_EQ(kernel->configure(grid, block)->launch(idata, odata), "");

  void* arg_ptrs[] = {&idata, &odata};

  int nrep = 10000;
  double dt_direct_ns = 1e99, dt_jitify_ns = 1e99;
  static const std::string kernel_inst =
      Template("my_kernel").instantiate(3, 4, type_of(*idata), type_of(*odata));
  for (int i = 0; i < nrep; ++i) {
    // Benchmark direct kernel launch.
    auto t0 = std::chrono::steady_clock::now();
    hip().ModuleLaunchKernel()(kernel->function(), grid.x, grid.y, grid.z,
                               block.x, block.y, block.z, 0, 0, arg_ptrs,
                               nullptr);
    auto dt = std::chrono::steady_clock::now() - t0;
    // Using the minimum is more robust than the average (though this test still
    // remains sensitive to the system environment and has been observed to fail
    // intermittently at a rate of <0.1%).
    dt_direct_ns = std::min(
        dt_direct_ns,
        (double)std::chrono::duration_cast<std::chrono::nanoseconds>(dt)
            .count());

    // Benchmark launch from cache.
    t0 = std::chrono::steady_clock::now();
    cache
        .get_kernel(
            // Note: It's faster to precompute this, but we want to test
            // worst-case perf.
            Template("my_kernel")
                .instantiate(3, 4, type_of(*idata), type_of(*odata)))
        ->configure(grid, block)
        ->launch(idata, odata);
    dt = std::chrono::steady_clock::now() - t0;
    dt_jitify_ns = std::min(
        dt_jitify_ns,
        (double)std::chrono::duration_cast<std::chrono::nanoseconds>(dt)
            .count());
  }
  double launch_time_direct_ns = dt_direct_ns;
  double launch_time_jitify_ns = dt_jitify_ns;
  // Ensure added latency is small.
  double tolerance_ns = 2500;  // 2.5us
  EXPECT_NEAR(launch_time_direct_ns, launch_time_jitify_ns, tolerance_ns);
}

class ScopeGuard {
  std::function<void()> func_;

 public:
  ScopeGuard(std::function<void()> func) : func_(std::move(func)) {}
  ~ScopeGuard() { func_(); }
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ScopeGuard(ScopeGuard&&) = delete;
  ScopeGuard& operator=(ScopeGuard&&) = delete;
};

inline bool remove_empty_dir(const char* path) {
#if defined(_WIN32) || defined(_WIN64)
  return ::_rmdir(path) == 0;
#else
  return ::rmdir(path) == 0;
#endif
}

TEST(Jitify2Test, ProgramCache) {
  static const char* const source = R"(
template <typename T>
__global__ void my_kernel(const T* __restrict__ idata, T* __restrict__ odata) {}
)";
  using key_type = uint32_t;
  size_t max_size = 2;
  static const char* const cache_path0 = "jitify2_test_cache";
  static const char* const cache_path = "jitify2_test_cache/subdir";
  ProgramCache<key_type> cache(max_size,
                               *Program("my_program", source)->preprocess(),
                               nullptr, cache_path);
  ScopeGuard scoped_cleanup_files([&] {
    cache.clear();
    remove_empty_dir(cache_path);
    remove_empty_dir(cache_path0);
  });

  auto check_hits = [&](size_t expected_hits, size_t expected_misses) {
    size_t num_hits, num_misses;
    cache.get_stats(&num_hits, &num_misses);
    EXPECT_EQ(num_hits, expected_hits);
    EXPECT_EQ(num_misses, expected_misses);
  };

  Kernel kernel;
  Template my_kernel("my_kernel");

  check_hits(0, 0);
  kernel = cache.get_kernel(/* key = */ 0, my_kernel.instantiate<float>());
  ASSERT_EQ(get_error(kernel), "");
  ASSERT_EQ(kernel->configure(1, 1)->launch(nullptr, nullptr), "");
  check_hits(0, 1);
  kernel = cache.get_kernel(/* key = */ 1, my_kernel.instantiate<double>());
  ASSERT_EQ(get_error(kernel), "");
  check_hits(0, 2);
  kernel = cache.get_kernel(/* key = */ 2, my_kernel.instantiate<int>());
  ASSERT_EQ(get_error(kernel), "");
  hipFunction_t function_int = kernel->function();
  check_hits(0, 3);
  cache.reset_stats();
  check_hits(0, 0);
  kernel = cache.get_kernel(/* key = */ 0, my_kernel.instantiate<float>());
  ASSERT_EQ(get_error(kernel), "");
  hipFunction_t function_float = kernel->function();
  check_hits(0, 1);
  kernel = cache.get_kernel(/* key = */ 2, my_kernel.instantiate<int>());
  ASSERT_EQ(get_error(kernel), "");
  EXPECT_EQ(kernel->function(), function_int);
  check_hits(1, 1);
  kernel = cache.get_kernel(/* key = */ 0, my_kernel.instantiate<float>());
  ASSERT_EQ(get_error(kernel), "");
  EXPECT_EQ(kernel->function(), function_float);
  check_hits(2, 1);
  LoadedProgram program =
      cache.get_program(/* key = */ 2, {my_kernel.instantiate<int>()});
  ASSERT_EQ(get_error(program), "");
  check_hits(3, 1);

  // Make sure cache dir was created.
  bool cache_path_is_dir;
  ASSERT_TRUE(jitify2::detail::path_exists(cache_path, &cache_path_is_dir));
  ASSERT_TRUE(cache_path_is_dir);
  // Make sure cache dir contains files.
  ASSERT_FALSE(remove_empty_dir(cache_path));
  // Now clear the cache.
  ASSERT_TRUE(cache.clear());
  EXPECT_EQ(cache.max_in_mem(), max_size);
  EXPECT_EQ(cache.max_files(), max_size);
  // Make sure cache dir still exists.
  ASSERT_TRUE(jitify2::detail::path_exists(cache_path, &cache_path_is_dir));
  ASSERT_TRUE(cache_path_is_dir);
  // Make sure cache dir is empty.
  ASSERT_TRUE(remove_empty_dir(cache_path));
  ASSERT_FALSE(jitify2::detail::path_exists(cache_path));

  max_size += 10;
  EXPECT_TRUE(cache.resize(max_size));
  EXPECT_EQ(cache.max_in_mem(), max_size);
  EXPECT_EQ(cache.max_files(), max_size);
  EXPECT_TRUE(cache.resize(max_size + 1, max_size + 2));
  EXPECT_EQ(cache.max_in_mem(), max_size + 1);
  EXPECT_EQ(cache.max_files(), max_size + 2);
}

TEST(Jitify2Test, ProgramCacheAutoKey) {
  static const char* const source = R"(
template <typename T>
__global__ void my_kernel(const T* __restrict__ idata, T* __restrict__ odata) {}
)";
  size_t max_size = 2;
  static const char* const cache_path0 = "jitify2_test_cache";
  static const char* const cache_path = "jitify2_test_cache/subdir";
  ProgramCache<> cache(max_size, *Program("my_program", source)->preprocess(),
                       nullptr, cache_path);
  ScopeGuard scoped_cleanup_files([&] {
    cache.clear();
    remove_empty_dir(cache_path);
    remove_empty_dir(cache_path0);
  });

  auto check_hits = [&](size_t expected_hits, size_t expected_misses) {
    size_t num_hits, num_misses;
    cache.get_stats(&num_hits, &num_misses);
    EXPECT_EQ(num_hits, expected_hits);
    EXPECT_EQ(num_misses, expected_misses);
  };

  Kernel kernel;
  Template my_kernel("my_kernel");

  check_hits(0, 0);
  kernel = cache.get_kernel(my_kernel.instantiate<float>());
  ASSERT_EQ(get_error(kernel), "");
  ASSERT_EQ(kernel->configure(1, 1)->launch(nullptr, nullptr), "");
  check_hits(0, 1);
  kernel = cache.get_kernel(my_kernel.instantiate<double>());
  ASSERT_EQ(get_error(kernel), "");
  check_hits(0, 2);
  kernel = cache.get_kernel(my_kernel.instantiate<int>());
  ASSERT_EQ(get_error(kernel), "");
  hipFunction_t function_int = kernel->function();
  check_hits(0, 3);
  cache.reset_stats();
  check_hits(0, 0);
  kernel = cache.get_kernel(my_kernel.instantiate<float>());
  ASSERT_EQ(get_error(kernel), "");
  hipFunction_t function_float = kernel->function();
  check_hits(0, 1);
  kernel = cache.get_kernel(my_kernel.instantiate<int>());
  ASSERT_EQ(get_error(kernel), "");
  EXPECT_EQ(kernel->function(), function_int);
  check_hits(1, 1);
  kernel = cache.get_kernel(my_kernel.instantiate<float>());
  ASSERT_EQ(get_error(kernel), "");
  EXPECT_EQ(kernel->function(), function_float);
  check_hits(2, 1);
  LoadedProgram program = cache.get_program({my_kernel.instantiate<int>()});
  ASSERT_EQ(get_error(program), "");
  check_hits(3, 1);

  // Make sure cache dir was created.
  bool cache_path_is_dir;
  ASSERT_TRUE(jitify2::detail::path_exists(cache_path, &cache_path_is_dir));
  ASSERT_TRUE(cache_path_is_dir);
  // Make sure cache dir contains files.
  ASSERT_FALSE(remove_empty_dir(cache_path));
  // Now clear the cache.
  ASSERT_TRUE(cache.clear());
  EXPECT_EQ(cache.max_in_mem(), max_size);
  EXPECT_EQ(cache.max_files(), max_size);
  // Make sure cache dir still exists.
  ASSERT_TRUE(jitify2::detail::path_exists(cache_path, &cache_path_is_dir));
  ASSERT_TRUE(cache_path_is_dir);
  // Make sure cache dir is empty.
  ASSERT_TRUE(remove_empty_dir(cache_path));
  ASSERT_FALSE(jitify2::detail::path_exists(cache_path));

  max_size += 10;
  EXPECT_TRUE(cache.resize(max_size));
  EXPECT_EQ(cache.max_in_mem(), max_size);
  EXPECT_EQ(cache.max_files(), max_size);
  EXPECT_TRUE(cache.resize(max_size + 1, max_size + 2));
  EXPECT_EQ(cache.max_in_mem(), max_size + 1);
  EXPECT_EQ(cache.max_files(), max_size + 2);
}

TEST(Jitify2Test, ProgramCacheFilenameSanitization) {
  static const char* const source = R"(__global__ void my_kernel() {})";
  const size_t max_size = 1;
  static const char* const cache_path = "jitify2_test_cache";
  // The filename is derived from the program name, so this checks that invalid
  // filename characters are automatically sanitized.
  ProgramCache<> cache(
      max_size, *Program("foo/bar/cat/dog\\:*?|<>", source)->preprocess(),
      nullptr, cache_path);
  ScopeGuard scoped_cleanup_files([&] {
    cache.clear();
    remove_empty_dir(cache_path);
  });
  *cache.get_kernel("my_kernel");
}

// TODO(HIP/AMD): Should be disabled with CUDA backend.
#ifdef __HIP_PLATFORM_AMD__
TEST(Jitify2Test, ProgramCacheTestLinkingExtraBitcode) {
  static const char* const source = R"(
  extern "C" __device__ void GENERIC_UNARY_OP(float*, float);
 
  template<typename T>
  __global__ void my_kernel(float *indata, float *outdata) {
    GENERIC_UNARY_OP(outdata, *indata);
  }
)";

  // Computes *C = a*a*a
  std::string amd_llvm_ir_udf = 
    R"'''(
; ModuleID = 'device_func-hip-amdgcn-amd-amdhsa-gfx90a.bc'
source_filename = "device_func.hip"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: convergent mustprogress noreturn nounwind
define weak void @__cxa_pure_virtual() #0 {
  call void @llvm.trap()
  unreachable
}

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #1

; Function Attrs: convergent mustprogress noreturn nounwind
define weak void @__cxa_deleted_virtual() #0 {
  call void @llvm.trap()
  unreachable
}

; Function Attrs: convergent mustprogress noinline nounwind
define hidden void @GENERIC_UNARY_OP(ptr %0, float %1) #2 {
  %3 = alloca ptr, align 8, addrspace(5)
  %4 = alloca float, align 4, addrspace(5)
  %5 = addrspacecast ptr addrspace(5) %3 to ptr
  %6 = addrspacecast ptr addrspace(5) %4 to ptr
  store ptr %0, ptr %5, align 8, !tbaa !7
  store float %1, ptr %6, align 4, !tbaa !11
  %7 = load float, ptr %6, align 4, !tbaa !11
  %8 = load float, ptr %6, align 4, !tbaa !11
  %9 = fmul contract float %7, %8
  %10 = load float, ptr %6, align 4, !tbaa !11
  %11 = fmul contract float %9, %10
  %12 = load ptr, ptr %5, align 8, !tbaa !7
  store float %11, ptr %12, align 4, !tbaa !11
  ret void
}

attributes #0 = { convergent mustprogress noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #1 = { cold noreturn nounwind }
attributes #2 = { convergent mustprogress noinline nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5}
!opencl.ocl.version = !{!6, !6, !6, !6, !6, !6, !6, !6, !6, !6}

!0 = !{i32 4, !"amdgpu_hostcall", i32 1}
!1 = !{i32 1, !"amdgpu_code_object_version", i32 500}
!2 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{!"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.0 23483 7208e8d15fbf218deb74483ea8c549c67ca4985e)"}
!6 = !{i32 2, i32 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"float", !9, i64 0}
    )'''";

  using key_type = uint32_t;
  size_t max_size = 2;
  static const char* const cache_path = "jitify2_test_cache";
  auto prog = Program("my_program", source)->preprocess();
  ProgramCache<key_type> cache(max_size,
                                *prog,
                                nullptr, cache_path);
  ScopeGuard scoped_cleanup_files([&] {
    cache.clear();
    remove_empty_dir(cache_path);
  });

  Template my_kernel("my_kernel");

  float* indata;
  float* outdata;
  CHECK_HIPRT(hipMalloc((void**)&indata, sizeof(float)));
  CHECK_HIPRT(hipMalloc((void**)&outdata, sizeof(float)));
  float inval = 3.0f;
  CHECK_HIPRT(hipMemcpy(indata, &inval, sizeof(float), hipMemcpyHostToDevice));

  auto kernel = cache.get_kernel(0, my_kernel.instantiate<float>(), {}, {}, {}, {}, &amd_llvm_ir_udf);
  ASSERT_EQ(get_error(kernel), "");
  ASSERT_EQ(kernel->configure(1, 1)->launch(indata, outdata), "");

  float outval = 0;
  CHECK_HIPRT(hipMemcpy(&outval, outdata, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIPRT(hipFree(outdata));
  CHECK_HIPRT(hipFree(indata));
 
  EXPECT_FLOAT_EQ(27.0, outval);
}
#endif

TEST(Jitify2Test, OfflinePreprocessing) {
  static const char* const extra_header_source = R"(
#pragma once
template <typename T>
__device__ T pointless_func(T x) {
  return x;
};)";
  size_t max_size = 10;
  // These variables come from the header generated by jitify_preprocess.
  ProgramCache<> cache(max_size, *jitify2_test_kernels_cu_jit,
                       jitify2_test_kernels_cu_headers_jit);
  enum { C = 123 };
  typedef float T;
  std::string kernel2_inst =
      Template("my_kernel2").instantiate<NonType<int, C>, T>();
  StringMap extra_headers = {{"my_header4.cuh", extra_header_source}};
  LoadedProgram program = cache.get_program(
      {"my_kernel1", kernel2_inst}, extra_headers, {"-includemy_header4.cuh"});
  ASSERT_EQ(get_error(program), "");

  T* indata;
  T* outdata;
  CHECK_HIPRT(hipMalloc((void**)&indata, sizeof(T)));
  CHECK_HIPRT(hipMalloc((void**)&outdata, sizeof(T)));
  T inval = 3.14159f;
  CHECK_HIPRT(hipMemcpy(indata, &inval, sizeof(T), hipMemcpyHostToDevice));

  dim3 grid(1), block(1);
  ASSERT_EQ(program->get_kernel("my_kernel1")
                ->configure(grid, block)
                ->launch(indata, outdata),
            "");
  ASSERT_EQ(program->get_kernel(kernel2_inst)
                ->configure(grid, block)
                ->launch(indata, outdata),
            "");

  T outval = 0;
  CHECK_HIPRT(hipMemcpy(&outval, outdata, sizeof(T), hipMemcpyDeviceToHost));
  CHECK_HIPRT(hipFree(outdata));
  CHECK_HIPRT(hipFree(indata));

  EXPECT_FLOAT_EQ(inval, outval);
}

TEST(Jitify2Test, Sha256) {
  EXPECT_EQ(jitify2::detail::sha256(""),
            "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855");
  EXPECT_EQ(jitify2::detail::sha256(std::string(1, '\0')),
            "6E340B9CFFB37A989CA544E6BB780A2C78901D3FB33738768511A30617AFA01D");
  EXPECT_EQ(jitify2::detail::sha256("a"),
            "CA978112CA1BBDCAFAC231B39A23DC4DA786EFF8147C4E72B9807785AFEE48BB");
  EXPECT_EQ(jitify2::detail::sha256("abc"),
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD");
  EXPECT_EQ(
      jitify2::detail::sha256("The quick brown fox jumps over the lazy dog."),
      "EF537F25C895BFA782526529A9B63D97AA631564D5D789C2B765448C8635FB6C");
  EXPECT_EQ(
      jitify2::detail::sha256("The quick brown fox jumps over the lazy dog."
                              "The quick brown fox jumps over the lazy dog."
                              "The quick brown fox jumps over the lazy dog."
                              "The quick brown fox jumps over the lazy dog."),
      "F5EA20F5EDD6871D72D699C143C524BF9CEC13D06E9FA5763614EE3BA708C63E");
}

TEST(Jitify2Test, PathBase) {
  EXPECT_EQ(jitify2::detail::path_base("foo/bar/2"), "foo/bar");
  EXPECT_EQ(jitify2::detail::path_base("foo/bar/2/"), "foo/bar/2");
  EXPECT_EQ(jitify2::detail::path_base("foo"), "");
  EXPECT_EQ(jitify2::detail::path_base("/"), "");
#if defined _WIN32 || defined _WIN64
  EXPECT_EQ(jitify2::detail::path_base("foo\\bar\\2"), "foo\\bar");
  EXPECT_EQ(jitify2::detail::path_base("foo\\bar\\2\\"), "foo\\bar\\2");
  EXPECT_EQ(jitify2::detail::path_base("foo"), "");
  EXPECT_EQ(jitify2::detail::path_base("\\"), "");
#endif
}

TEST(Jitify2Test, PathJoin) {
  EXPECT_EQ(jitify2::detail::path_join("foo/bar", "2/1"), "foo/bar/2/1");
  EXPECT_EQ(jitify2::detail::path_join("foo/bar/", "2/1"), "foo/bar/2/1");
  EXPECT_EQ(jitify2::detail::path_join("foo/bar", "/2/1"), "");
#if defined _WIN32 || defined _WIN64
  EXPECT_EQ(jitify2::detail::path_join("foo\\bar", "2\\1"), "foo\\bar/2\\1");
  EXPECT_EQ(jitify2::detail::path_join("foo\\bar\\", "2\\1"), "foo\\bar\\2\\1");
  EXPECT_EQ(jitify2::detail::path_join("foo\\bar", "\\2\\1"), "");
#endif
}

TEST(Jitify2Test, PathSimplify) {
  EXPECT_EQ(jitify2::detail::path_simplify(""), "");
  EXPECT_EQ(jitify2::detail::path_simplify("/"), "/");
  EXPECT_EQ(jitify2::detail::path_simplify("//"), "/");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/bar"), "/foo/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/bar"), "foo/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/./bar"), "/foo/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/./bar"), "foo/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/../bar"), "/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/../bar"), "bar");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/cat/../../bar"), "/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/cat/../../bar"), "bar");
  EXPECT_EQ(jitify2::detail::path_simplify("/./bar"), "/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("./bar"), "bar");
  EXPECT_EQ(jitify2::detail::path_simplify("../bar"), "../bar");
  EXPECT_EQ(jitify2::detail::path_simplify("../../bar"), "../../bar");
  EXPECT_EQ(jitify2::detail::path_simplify("../.././bar"), "../../bar");
  EXPECT_EQ(jitify2::detail::path_simplify(".././../bar"), "../../bar");
  EXPECT_EQ(jitify2::detail::path_simplify("./../../bar"), "../../bar");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/bar/.."), "/foo");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/bar/.."), "foo");
  EXPECT_EQ(jitify2::detail::path_simplify("//foo///..////bar"), "/bar");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/"), "foo/");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/"), "/foo/");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/bar/"), "foo/bar/");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/bar/"), "/foo/bar/");
  EXPECT_EQ(jitify2::detail::path_simplify("foo/../bar/"), "bar/");
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/../bar/"), "/bar/");
  EXPECT_EQ(jitify2::detail::path_simplify("/../foo"), "");    // Invalid path
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/../../bar"),  // Invalid path
            "");
  EXPECT_EQ(jitify2::detail::path_simplify("/.."), "");         // Invalid path
  EXPECT_EQ(jitify2::detail::path_simplify("/foo/../.."), "");  // Invalid path
#if defined _WIN32 || defined _WIN64
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\)"), R"(\)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\\)"), R"(\)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\foo\bar)"), R"(\foo\bar)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(foo\bar)"), R"(foo\bar)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\foo\.\bar)"), R"(\foo\bar)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(foo\.\bar)"), R"(foo\bar)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\foo\..\bar)"), R"(\bar)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(foo\..\bar)"), R"(bar)");

  EXPECT_EQ(jitify2::detail::path_simplify(R"(\foo/.\bar)"), R"(\foo/bar)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\foo/.\bar\./cat)"),
            R"(\foo/bar\cat)");
  EXPECT_EQ(jitify2::detail::path_simplify(R"(\foo/.\bar\../cat)"),
            R"(\foo/cat)");
#endif
}

TEST(Jitify2Test, Program) {
  static const char* const name = "my_program";
  static const char* const source = "/* empty source */";
  static const char* const header_name = "my_header";
  static const char* const header_source = "/* empty header */";
  Program program;
  ASSERT_EQ(static_cast<bool>(program), false);
  EXPECT_EQ(program.error(), "Uninitialized");
  EXPECT_THROW(*program, std::runtime_error);
  program = Program(name, source, {{header_name, header_source}});
  ASSERT_EQ(get_error(program), "");
  EXPECT_THROW(program.error(), std::runtime_error);
  EXPECT_EQ(program->name(), name);
  EXPECT_EQ(program->source(), source);
  EXPECT_EQ(program->header_sources().size(), size_t(1));
  ASSERT_EQ(program->header_sources().count(header_name), size_t(1));
  EXPECT_EQ(program->header_sources().at(header_name), header_source);
}

bool contains(const std::string& src, const std::string& target,
              const char* varname) {
  bool result = src.find(target) != std::string::npos;
  if (!result) {
    std::cerr << "--- BEGIN STRING " << varname << " ---\n"
              << src << "\n--- END STRING " << varname << " ---" << std::endl;
  }
  return result;
}

TEST(Jitify2Test, PreprocessedProgram) {
  // Tests source patching, header extraction, use of builtin headers, and basic
  // PreprocessedProgram API functionality.
  static const char* const name = "my_program";
  static const char* const source = R"(
#include <my_header1.cuh>
__global__ void my_kernel() {}
)";
  static const char* const header_name = "my_header1.cuh";
  Program program(name, source);
  ASSERT_EQ(get_error(program), "");
  PreprocessedProgram preprog = program->preprocess();
  ASSERT_EQ(static_cast<bool>(preprog), false);
  EXPECT_TRUE(CONTAINS(preprog.error(), "File not found"));
  preprog = program->preprocess({"-Iexample_headers"}, {"-lfoo"});
  ASSERT_EQ(get_error(preprog), "");
  EXPECT_EQ(preprog->name(), name);
  EXPECT_EQ(preprog->header_sources().count(header_name), size_t(1));
  EXPECT_TRUE(
      NOT_CONTAINS(preprog->remaining_compiler_options(), "-Iexample_headers"));
  EXPECT_EQ(preprog->remaining_linker_options(), StringVec({"-lfoo"}));
  EXPECT_NE(preprog->header_log(), "");
  EXPECT_EQ(preprog->compile_log(), "");
}

TEST(Jitify2Test, CompiledProgram) {
  // Tests compilation, lowered name lookup, and basic CompiledProgram API
  // functionality.
  static const char* const name = "my_program";
  static const char* const source = R"(
template <typename T>
__global__ void my_kernel() {}
)";
  static const char* const instantiation = "my_kernel<float>";
  static const char* const lowered_name = "_Z9my_kernelIfEvv";
  Program program(name, source);
  ASSERT_EQ(get_error(program), "");
  PreprocessedProgram preprog = program->preprocess();
  ASSERT_EQ(get_error(preprog), "");
  // todo(hip): Add support for "--remove-unused-globals" flag
  CompiledProgram compiled = preprog->compile(instantiation, {}, {}, {"-lfoo"});
  ASSERT_EQ(get_error(compiled), "");
  EXPECT_NE(compiled->binary(), "");
  EXPECT_EQ(compiled->lowered_name_map().size(), size_t(1));
  ASSERT_EQ(compiled->lowered_name_map().count(instantiation), size_t(1));
  EXPECT_EQ(compiled->lowered_name_map().at(instantiation), lowered_name);
  std::unordered_multiset<std::string> linker_options;
  linker_options.insert(compiled->remaining_linker_options().begin(),
                        compiled->remaining_linker_options().end());
  EXPECT_EQ(linker_options.count("-lfoo"), 1);
  EXPECT_EQ(compiled->log(), "");
}

TEST(Jitify2Test, ConstantMemory) {
  static const char* const source = R"(
__constant__ int a;
__device__ int d;
namespace b { __constant__ int a; __device__ int d; }
namespace c { namespace b { __constant__ int a; __device__ int d; } }
namespace x { __constant__ int a = 3; __device__ int d = 7; }
namespace y { __constant__ int a[] = {4, 5}; __device__ int d[] = {8, 9}; }
namespace z { template <typename T> __constant__ T tv = 10; }

__global__ void constant_test(int* x) {
  x[0] = a;
  x[1] = b::a;
  x[2] = c::b::a;
  x[3] = d;
  x[4] = b::d;
  x[5] = c::b::d;
  x[6] = x::a;
  x[7] = x::d;
  x[8] = y::a[0];
  x[9] = y::a[1];
  x[10] = y::d[0];
  x[11] = y::d[1];
})";

  dim3 grid(1), block(1);
  {  // Test __constant__ look up in kernel using different namespaces.
    Kernel kernel =
        Program("constmem_program", source)
            ->preprocess({"-std=c++14"})
            // TODO: Use z::tv<float> in tests below.
            // todo(HIP): we are adding the name expressions manually, as they
            // cannot be easily extracted from LLVM bitcode/ISA using hiprtc
            ->get_kernel(
                "constant_test",
                {"&z::tv<float>", "&x::a", "&x::d", "&y::a", "&y::d", "&a",
                 "&b::a", "&c::b::a", "&d", "&b::d", "&c::b::d"});
    const LoadedProgramData& program = kernel->program();
    int dval;
    ASSERT_EQ(program.get_global_value("x::a", &dval), "");
    EXPECT_EQ(dval, 3);
    ASSERT_EQ(program.get_global_value("x::d", &dval), "");
    EXPECT_EQ(dval, 7);
    int darr[2];
    ASSERT_EQ(program.get_global_data("y::a", &darr[0], 2), "");
    EXPECT_EQ(darr[0], 4);
    EXPECT_EQ(darr[1], 5);
    ASSERT_EQ(program.get_global_value("y::d", &darr), "");
    EXPECT_EQ(darr[0], 8);
    EXPECT_EQ(darr[1], 9);
    int inval[] = {2, 4, 8, 12, 14, 18, 22, 26, 30, 34, 38, 42};
    constexpr int n_const = sizeof(inval) / sizeof(int);
    ASSERT_EQ(program.set_global_value("a", inval[0]), "");
    ASSERT_EQ(program.set_global_value("b::a", inval[1]), "");
    ASSERT_EQ(program.set_global_value("c::b::a", inval[2]), "");
    ASSERT_EQ(program.set_global_value("d", inval[3]), "");
    ASSERT_EQ(program.set_global_value("b::d", inval[4]), "");
    ASSERT_EQ(program.set_global_value("c::b::d", inval[5]), "");
    ASSERT_EQ(program.set_global_value("x::a", inval[6]), "");
    ASSERT_EQ(program.set_global_value("x::d", inval[7]), "");
    ASSERT_EQ(program.set_global_data("y::a", &inval[8], 2), "");
    int inarr[] = {inval[10], inval[11]};
    ASSERT_EQ(program.set_global_value("y::d", inarr), "");
    int* outdata;
    CHECK_HIPRT(hipMalloc((void**)&outdata, n_const * sizeof(int)));
    ASSERT_EQ(kernel->configure(grid, block)->launch(outdata), "");
    CHECK_HIPRT(hipDeviceSynchronize());
    int outval[n_const];
    CHECK_HIPRT(
        hipMemcpy(outval, outdata, sizeof(outval), hipMemcpyDeviceToHost));
    for (int i = 0; i < n_const; i++) {
      EXPECT_EQ(inval[i], outval[i]);
    }
    CHECK_HIPRT(hipFree(outdata));
  }
}

TEST(Jitify2Test, InvalidPrograms) {
  // OK.
  EXPECT_EQ(get_error(Program("empty_program", "")->preprocess()),
            "Compilation failed: HIPRTC_ERROR_INVALID_INPUT\nCompiler options: "
            "\"-std=c++11\"\n");
  // OK.
  EXPECT_EQ(
      get_error(Program("found_header", "#include <cstdio>")->preprocess()),
      "");
  // Not OK.
  EXPECT_NE(
      get_error(
          Program("missing_header", "#include <cantfindme>")->preprocess()),
      "");
  // Not OK.
  EXPECT_NE(get_error(Program("bad_program", "NOT HIP C!")->preprocess()), "");
}

TEST(Jitify2Test, LinkMultiplePrograms) {
  static const char* const source1 = R"(
__constant__ int c = 5;
__device__ int d = 7;
__device__ int f(int i) { return i + 11; }
)";

  static const char* const source2 = R"(
extern __constant__ int c;
extern __device__ int d;
extern __device__ int f(int);
__global__ void my_kernel(int* data) {
  *data = f(*data + c + d);
}
)";

  CompiledProgram program1 = Program("linktest_program1", source1)
                                 ->preprocess({"-fgpu-rdc"})
                                 ->compile();
  CompiledProgram program2 = Program("linktest_program2", source2)
                                 ->preprocess({"-fgpu-rdc"})
                                 ->compile("my_kernel");
  // TODO: Consider allowing refs not ptrs for programs, and also addding a
  //         get_kernel() shortcut method to LinkedProgram.
  Kernel kernel = LinkedProgram::link({&program1, &program2})
                      ->load()
                      ->get_kernel("my_kernel");
  int* d_data;
  CHECK_HIPRT(hipMalloc((void**)&d_data, sizeof(int)));
  int h_data = 3;
  CHECK_HIPRT(hipMemcpy(d_data, &h_data, sizeof(int), hipMemcpyHostToDevice));
  ASSERT_EQ(kernel->configure(1, 1)->launch(d_data), "");
  CHECK_HIPRT(hipMemcpy(&h_data, d_data, sizeof(int), hipMemcpyDeviceToHost));
  EXPECT_EQ(h_data, 26);
  CHECK_HIPRT(hipFree(d_data));
}

// TODO(HIP/AMD): This test is currently not working due to a bug in hiprtc with
// ROCm >=5.5. It should run when ticket
// SWDEV-415448 has been resolved (with
// ROCm 5.7) TEST(Jitify2Test, LinkExternalFiles) {
//   static const char* const source1 = R"(
//__constant__ int c = 5;
//__device__ int d = 7;
//__device__ int f(int i) { return i + 11; })";

//  static const char* const source2 = R"(
// extern __constant__ int c;
// extern __device__ int d;
// extern __device__ int f(int);
//__global__ void my_kernel(int* data) {
//  *data = f(*data + c + d);
//})";

//  // Ensure temporary file is deleted at the end.
//  std::unique_ptr<const char, int (*)(const char*)> bc_filename(
//      "example_headers/linktest.bc", std::remove);
//  {
//    std::ofstream bc_file(bc_filename.get());
//    bc_file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
//    bc_file << Program("linktest_program1", source1)
//                    ->preprocess({"-fgpu-rdc"})
//                    ->compile()
//                    ->bitcode();
//  }
//  const std::vector<std::string> linker_options = {"-Lexample_headers",
//                                                      "-llinktest.bc"};
//    Kernel kernel = Program("linktest_program2", source2)
//                        ->preprocess({"-fgpu-rdc"}, linker_options)
//                        ->get_kernel("my_kernel");
//    int* d_data;
//    CHECK_HIPRT(hipMalloc((void**)&d_data, sizeof(int)));
//    int h_data = 3;
//    CHECK_HIPRT(
//        hipMemcpy(d_data, &h_data, sizeof(int), hipMemcpyHostToDevice));
//    ASSERT_EQ(kernel->configure(1, 1)->launch(d_data), "");
//    CHECK_HIPRT(
//        hipMemcpy(&h_data, d_data, sizeof(int), hipMemcpyDeviceToHost));
//    EXPECT_EQ(h_data, 26);
//    CHECK_HIPRT(hipFree(d_data));
//}

namespace a {
__host__ __device__ int external_device_func(int i) { return i + 1; }
}  // namespace a

TEST(Jitify2Test, ClassKernelArg) {
  static const char* const source = R"(
#include "example_headers/class_arg_kernel.cuh"
)";

  int h_data;
  int* d_data;
  CHECK_HIPRT(hipMalloc((void**)&d_data, sizeof(int)));

  PreprocessedProgram preprog =
      Program("class_kernel_arg_program", source)->preprocess();
  ConfiguredKernel configured_kernel =
      preprog->get_kernel(Template("class_arg_kernel").instantiate<Arg>())
          ->configure(1, 1);

  {  // Test that we can pass an arg object to a kernel.
    Arg arg(-1);
    ASSERT_EQ(configured_kernel->launch(d_data, arg), "");
    CHECK_HIPRT(hipDeviceSynchronize());
    CHECK_HIPRT(hipMemcpy(&h_data, d_data, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(arg.x, h_data);
  }

  {  // Test that we can pass an arg object rvalue to a kernel.
    int value = -2;
    ASSERT_EQ(configured_kernel->launch(d_data, Arg(value)), "");
    CHECK_HIPRT(hipDeviceSynchronize());
    CHECK_HIPRT(hipMemcpy(&h_data, d_data, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(value, h_data);
  }

  {  // Test that we can pass an arg object reference to a kernel.
    std::unique_ptr<Arg> arg(new Arg(-3));
    // References are passed as pointers since refernces are just pointers from
    // an ABI point of view.
    ASSERT_EQ(
        preprog->get_kernel(Template("class_arg_ref_kernel").instantiate<Arg>())
            ->configure(1, 1)
            ->launch(d_data, arg.get()),
        "");
    CHECK_HIPRT(hipMemcpy(&h_data, d_data, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(arg->x, h_data);
  }

  {  // Test that we can pass an arg object reference to a kernel
    std::unique_ptr<Arg> arg(new Arg(-4));
    ASSERT_EQ(
        preprog->get_kernel(Template("class_arg_ptr_kernel").instantiate<Arg>())
            ->configure(1, 1)
            ->launch(d_data, arg.get()),
        "");
    CHECK_HIPRT(hipMemcpy(&h_data, d_data, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(arg->x, h_data);
  }

  CHECK_HIPRT(hipFree(d_data));
}

TEST(Jitify2Test, GetAttribute) {
  static const char* const source = R"(
__global__ void get_attribute_kernel(int* out, const int* in) {
  __shared__ int buffer[4096];
  buffer[threadIdx.x] = in[threadIdx.x];
  __syncthreads();
  out[threadIdx.y] = buffer[threadIdx.x];
}
)";

  // Checks that we can get function attributes.
  int attrval;
  ASSERT_EQ(Program("get_attribute_program", source)
                ->preprocess()
                ->get_kernel("get_attribute_kernel")
                ->get_attribute(HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, &attrval),
            "");
  EXPECT_EQ(attrval, 4096 * (int)sizeof(int));
}

TEST(Jitify2Test, SetAttribute) {
  static const char* const source = R"(
__global__ void set_attribute_kernel(int* out, int* in) {
  extern __shared__ int buffer[];
  buffer[threadIdx.x] = in[threadIdx.x];
  __syncthreads();
  out[threadIdx.y] = buffer[threadIdx.x];
}
)";

  int* in;
  CHECK_HIPRT(hipMalloc((void**)&in, sizeof(int)));
  int* out;
  CHECK_HIPRT(hipMalloc((void**)&out, sizeof(int)));

  // Query the maximum supported shared bytes per block.
  // Todo(HIP): HIP does not support hipDeviceAttributeSharedMemPerBlockOptin.
  // Use hipDeviceAttributeMaxSharedMemoryPerBlock instead?
  hipDevice_t device;
  CHECK_HIP(hip().DeviceGet()(&device, 0));
  int shared_bytes;
  CHECK_HIP(hip().DeviceGetAttribute()(
      &shared_bytes, hipDeviceAttributeMaxSharedMemoryPerBlock, device));

  Kernel kernel = Program("set_attribute_program", source)
                      ->preprocess()
                      ->get_kernel("set_attribute_kernel");
  ASSERT_EQ(kernel->set_attribute(
                HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes),
            "");

  dim3 grid(1), block(1);
  ASSERT_EQ(kernel->configure(grid, block, (unsigned int) shared_bytes)->launch(out, in), "");

  CHECK_HIPRT(hipFree(out));
  CHECK_HIPRT(hipFree(in));
}

TEST(Jitify2Test, ArchFlags) {
  // HIP: We have made some changes to the original test because:
  // 1. We dont have __HIP_ARCH__ and we cannot get the arch name on the device.
  // We get the current arch and pass it to the kernel.
  // 2. We first need to register the global variable (i.e., arch) before we can
  // retrieve its value with get_global_value.
  static const char* const source = R"(
__device__ char* arch = nullptr;
__global__ void my_kernel( char* data) {
  arch = data;
}
)";
  std::string current_arch = get_current_device_arch();
  char* arch_char;

  // Test default behavior (automatic architecture detection).
  PreprocessedProgram preprocessed =
      Program("arch_flags_program", source)->preprocess();

  Kernel kernel = preprocessed->get_kernel("my_kernel", {"&arch"});
  const LoadedProgramData& loaded_program = kernel->program();
  CompiledProgram program = preprocessed->compile();

  ASSERT_EQ(kernel->configure(1, 1)->launch(current_arch.c_str()), "");
  ASSERT_EQ(loaded_program.get_global_value("arch", &arch_char), "");
  std::string arch(arch_char);
  EXPECT_EQ(arch, current_arch);

  // Test explicit architecture (compile to BC).
  program = preprocessed->compile("", {}, {"--offload-arch=gfx90a"});
  ASSERT_GT(program->binary().size(), 0);
  ASSERT_GT(program->bitcode().size(), 0);

  auto expect_bitcode_size_if_available = [](size_t bitcode_size) {
    if (jitify2::hiprtc().GetBitcode()) {
      EXPECT_GT(bitcode_size, 0);
    } else {
      EXPECT_EQ(bitcode_size, 0);
    }
  };

  // Test explicit real architecture (may compile directly to bitcode).
  program = preprocessed->compile("", {}, {"--offload-arch", current_arch});
  ASSERT_GT(program->binary().size(), 0);
  expect_bitcode_size_if_available(program->bitcode().size());
  // ASSERT_EQ(program->link()->load()->get_global_value("arch", &arch), "");
  EXPECT_EQ(arch, current_arch);

  // Test automatic virtual architecture (compile to BC).
  program = preprocessed->compile("", {}, {"--offload-arch", "gfx."});
  EXPECT_GT(program->binary().size(), 0);
  EXPECT_GT(program->bitcode().size(), 0);
  // ASSERT_EQ(program->link()->load()->get_global_value("arch", &arch), "");
  EXPECT_EQ(arch, current_arch);

  // Test automatic real architecture (may compile directly to bitcode).
  program = preprocessed->compile("", {}, {"--offload-arch=gfx."});
  EXPECT_GT(program->binary().size(), 0);
  expect_bitcode_size_if_available(program->bitcode().size());
  // ASSERT_EQ(program->link()->load()->get_global_value("arch", &arch), "");
  EXPECT_EQ(arch, current_arch);

  // Test that preprocessing and compilation use separate arch flags.
  program = Program("arch_flags_program", source)
                ->preprocess({"--offload-arch=gfx90a"})
                ->compile("", {}, {"--offload-arch=gfx."});
  EXPECT_GT(program->binary().size(), 0);
  expect_bitcode_size_if_available(program->bitcode().size());
  // ASSERT_EQ(program->link()->load()->get_global_value("arch", &arch), "");
  // ASSERT_EQ(loaded_program.get_global_value("arch", &arch_char), "");
  EXPECT_EQ(arch, current_arch);

  // Test that multiple architectures can be specified for preprocessing.
  program = Program("arch_flags_program", source)
                ->preprocess({"--offload-arch=gfx90a", "--offload-arch=gfx908",
                              "--offload-arch=gfx906"})
                ->compile("", {}, {"--offload-arch=gfx."});
  EXPECT_GT(program->binary().size(), 0);
  EXPECT_GT(program->bitcode().size(), 0);
  ASSERT_EQ(get_error(program), "");

  // HIPRTC does not support the flag maxrregcount
  // Test that certain compiler options are automatically passed to the linker.
  LinkedProgram linked =
      Program("arch_flags_program", source)
          ->preprocess({"-g"}) 
          ->compile()
          ->link();
  ASSERT_EQ(get_error(linked), "");
  std::unordered_multiset<std::string> linker_options(
      linked->linker_options().begin(), linked->linker_options().end());
  EXPECT_EQ(linker_options.count("-g"), 1);
}

struct Base {
  virtual ~Base() {}
};
template <typename T>
struct Derived : public Base {};

TEST(Jitify2Test, Reflection) {
  static const char* const source = R"(
struct Base { virtual ~Base() {} };
template <typename T>
struct Derived : public Base {};
template <typename T>
__global__ void type_kernel() {}
template <unsigned short N>
__global__ void nontype_kernel() {}
)";

  PreprocessedProgram preprog =
      Program("reflection_program", source)->preprocess();

  Template type_kernel("type_kernel");

#define JITIFY_TYPE_REFLECTION_TEST(T)                                   \
  EXPECT_EQ(                                                             \
      preprog->get_kernel(type_kernel.instantiate<T>())->lowered_name(), \
      preprog->get_kernel(type_kernel.instantiate({#T}))->lowered_name())

  // todo(hip): when using template based reflection, instantiate<T> does
  // reorder expressions such as "const volatile float" -> "float const
  // volatile"
  JITIFY_TYPE_REFLECTION_TEST(float const volatile);
  JITIFY_TYPE_REFLECTION_TEST(float* const volatile);
  JITIFY_TYPE_REFLECTION_TEST(float const volatile&);
  // JITIFY_TYPE_REFLECTION_TEST(Base * (const volatile float));  //todo(hip):
  // reflection is not working properly currently
  JITIFY_TYPE_REFLECTION_TEST(float const volatile[4]);

#undef JITIFY_TYPE_REFLECTION_TEST

  typedef Derived<float> derived_type;
  const Base& base = derived_type();
  EXPECT_EQ(preprog->get_kernel(type_kernel.instantiate(instance_of(base)))
                ->lowered_name(),
            preprog->get_kernel(type_kernel.instantiate<derived_type>())
                ->lowered_name());

  Template nontype_kernel("nontype_kernel");

#define JITIFY_NONTYPE_REFLECTION_TEST(N)                                 \
  EXPECT_EQ(                                                              \
      preprog->get_kernel(nontype_kernel.instantiate(N))->lowered_name(), \
      preprog->get_kernel(nontype_kernel.instantiate({#N}))->lowered_name())

  // JITIFY_NONTYPE_REFLECTION_TEST(7); //todo(hip): this test is not working
  // due to a bug in hiprtc, it may be resolved with ticket
  // SWDEV-379212
  // JITIFY_NONTYPE_REFLECTION_TEST('J'); //todo(hip): this test is not working
  // due to a bug in hiprtc, it may be resolved with ticket
  // SWDEV-379212

#undef JITIFY_NONTYPE_REFLECTION_TEST
}

TEST(Jitify2Test, BuiltinNumericLimitsHeader) {
  static const char* const source = R"(
#include <limits>
struct MyType {};
namespace std {
template<> class numeric_limits<MyType> {
 public:
  static MyType __host__ __device__ min() { return {}; }
  static MyType __host__ __device__ max() { return {}; }
};
}  // namespace std
template <typename T>
__global__ void my_kernel(T* data) {
  data[0] = std::numeric_limits<T>::min();
  data[1] = std::numeric_limits<T>::max();
}
)";
  PreprocessedProgram preprog =
      Program("builtin_numeric_limits_program", source)->preprocess();
  for (const auto& type :
       {"float", "double", "char", "signed char", "unsigned char", "short",
        "unsigned short", "int", "unsigned int", "long", "unsigned long",
        "long long", "unsigned long long", "MyType"}) {
    std::string kernel_inst = Template("my_kernel").instantiate(type);
    Kernel kernel =
        preprog->compile(kernel_inst)->link()->load()->get_kernel(kernel_inst);
    (void)kernel;
  }
}

TEST(Jitify2Test, LibHipCxx) {
  // HIP: Following headers are not supported in libhipcxx currently: barrier,
  // latch, semaphore
  // TODO(HIP): The header functional does not pass preprocessing currently, we will likely need to fix this in libhipcxx.
  // Test that each libhipcxx header can be compiled on its
  // own.
  for (const std::string header :
       {/*"barrier", "latch", "semaphore"*/
         "atomic", "cassert", "cfloat", "chrono", "climits", "cstddef",
        "cstdint", "ctime", "ratio", "type_traits", "utility",
        "limits" /*, functional*/}) {
    std::string source =
        "#include <hip/std/" + header + ">\n__global__ void my_kernel() {}";
    // Note: The -arch flag here is required because "HIP atomics are
    // only supported for gfx90a "TODO(HIP): gfx908?" and up."
    Program("libhipcxx_program", source)
        ->preprocess({"-I" LIBHIPCXX_INC_DIR, "--offload-arch=gfx90a",
                      /*"-no-builtin-headers", "-no-preinclude-workarounds",
                      "-no-system-headers-workaround",
                      "-no-replace-pragma-once"*/})
        ->get_kernel("my_kernel");
  }
  // WAR for bug in hip/std/limits that is missing include hip/std/climits.
  static const char* const source = R"(
#include <hip/std/climits>
#include <hip/std/limits>
__global__ void my_kernel() {}
)";
  Program("libhipcxx_program", source)
      ->preprocess({"-I" LIBHIPCXX_INC_DIR, "--offload-arch=gfx90a",
                    /*"-no-builtin-headers", "-no-preinclude-workarounds",
                    "-no-system-headers-workaround", "-no-replace-pragma-once"*/})
      ->get_kernel("my_kernel");
}

TEST(Jitify2Test, Minify) {
  static const char* const name = "my_program";
  // This source is intentionally tricky to parse so that it stresses the
  // minification algorithm.
  static const std::string source = R"(
 //#define FOO foo
 //#define BAR(call)                             \
 //  do {                                        \
 //    call;                                     \
 //  } while (0)

 #ifndef __HIPCC_RTC__
     #define FOOBAR
     #define BARFOO
 #else
     #define MY_CHAR_BIT 8
     #define __MY_CHAR_UNSIGNED__ ('\xff' > 0) // CURSED
    #if __MY_CHAR_UNSIGNED__
        #define MY_CHAR_MIN 0
         #define MY_CHAR_MAX UCHAR_MAX
     #else
         #define MY_CHAR_MIN SCHAR_MIN
         #define MY_CHAR_MAX SCHAR_MAX
     #endif
 #endif
 /*
 This will
 all be
 "trickily"
 removed
 hopefully.*/

 const char* const foo = R"foo(abc\def
 ghi"')foo";  // )'

 //todo(HIP): hiprtc yields redefinition errors, if these headers are included
   #include <iterator>  // Here's a comment
   #include <tuple>  // Here's another comment

 const char* const linecont_str = "line1 \
 line2";
 const char c = '\xff';

 //todo(HIP): hiprtc yields redefinition errors, if these headers are included;
 // add correct HIP_VERSION when this is supported
 //#include <hip/hip_runtime.h>
 //#if HIP_VERSION >= 60000000 
 // CUB headers can be tricky to parse.
 //#include <hipcub/block/block_load.cuh>
 //#include <hipcub/block/block_radix_sort.cuh>
 //#include <hipcub/block/block_reduce.cuh>
 //#include <hipcub/block/block_store.cuh>
 //#endif  // HIP_VERSION >= 60000000 

 #include "example_headers/my_header1.cuh"
 __global__ void my_kernel() {}
 )";
  PreprocessedProgram preprog =
      Program(name, source)->preprocess({"-I" HIPCUB_DIR, "-I" HIP_INC_DIR});
  ASSERT_EQ(get_error(preprog), "");
  CompiledProgram compiled = preprog->compile();
  ASSERT_EQ(get_error(compiled), "");
  std::string orig_binary = compiled->binary();

  preprog = Program(name, source)
                ->preprocess({"-I" HIPCUB_DIR, "-I" HIP_INC_DIR, "--minify"});
  ASSERT_EQ(get_error(preprog), "");
  EXPECT_LT(preprog->source().size(), source.size());
  compiled = preprog->compile();
  ASSERT_EQ(get_error(compiled), "");
  ASSERT_EQ(compiled->binary(), orig_binary);
}

int main(int argc, char** argv) {
  hipSetDevice(0);
  // Initialize the driver context (avoids "initialization error"/"context is
  // destroyed").
  hipFree(0);
  ::testing::InitGoogleTest(&argc, argv);
  // Test order is actually undefined, so we use filters to force the
  // AssertHeader test to run last.
  ::testing::GTEST_FLAG(filter) += ":-Jitify2Test.AssertHeader";
  int result = RUN_ALL_TESTS();
  ::testing::GTEST_FLAG(filter) = "Jitify2Test.AssertHeader";
  return result | RUN_ALL_TESTS();
}
