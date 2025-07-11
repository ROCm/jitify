// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#define CUDART_VERSION 0
#define CUDA_VERSION 0

#ifndef nvrtcProgram
#define nvrtcProgram hiprtcProgram
#endif
#ifndef nvrtcResult
#define nvrtcResult hiprtcResult
#endif
#ifndef cudaError_t
#  define cudaError_t hipError_t
#endif
#ifndef cudaSuccess
#  define cudaSuccess hipSuccess
#endif
#ifndef NVRTC_SUCCESS
#define NVRTC_SUCCESS HIPRTC_SUCCESS
#endif
#ifndef cudaSuccess
#  define cudaSuccess hipSuccess
#endif
#ifndef CUDA_SUCCESS
#  define CUDA_SUCCESS hipSuccess
#endif
#ifndef CUresult
#  define CUresult hipError_t
#endif
#ifndef CUdevice
#  define CUdevice hipDevice_t
#endif
#ifndef CUcontext
#  define CUcontext hipCtx_t
#endif
#ifndef CUmodule
#  define CUmodule hipModule_t
#endif
#ifndef CUfunction
#  define CUfunction hipFunction_t
#endif
#ifndef nvrtcCreateProgram
#  define nvrtcCreateProgram hiprtcCreateProgram
#endif
#ifndef nvrtcDestroyProgram
#  define nvrtcDestroyProgram hiprtcDestroyProgram
#endif
#ifndef nvrtcGetCUBIN 
#  define nvrtcGetCUBIN hiprtcGetCode
#endif
#ifndef nvrtcGetCUBINSize
#  define nvrtcGetCUBINSize hiprtcGetCodeSize
#endif
#ifndef nvrtcGetLoweredName
#  define nvrtcGetLoweredName hiprtcGetLoweredName
#endif
#ifndef nvrtcGetNVVM
#  define nvrtcGetNVVM hiprtcGetBitcode
#endif 
#ifndef nvrtcGetNVVMSize
#  define nvrtcGetNVVMSize hiprtcGetBitcodeSize
#endif
#ifndef nvrtcGetBitcode
#  define nvrtcGetBitcode hiprtcGetBitcode
#endif
#ifndef nvrtcGetBitcodeSize
#  define nvrtcGetBitcodeSize hiprtcGetBitcodeSize
#endif
#ifndef nvrtcGetProgramLog
#  define nvrtcGetProgramLog hiprtcGetProgramLog
#endif
#ifndef nvrtcGetProgramLogSize
#  define nvrtcGetProgramLogSize hiprtcGetProgramLogSize
#endif
#ifndef nvrtcCompileProgram
#  define nvrtcCompileProgram hiprtcCompileProgram
#endif
#ifndef nvrtcGetErrorString
#  define nvrtcGetErrorString hiprtcGetErrorString
#endif
#ifndef cuLaunchKernel
#  define cuLaunchKernel hipModuleLaunchKernel
#endif
#ifndef cuInit
#  define cuInit hipInit
#endif
#ifndef cuCtxCreate
#  define cuCtxCreate hipCtxCreate
#endif
#ifndef cuModuleGetFunction  
#  define cuModuleGetFunction hipModuleGetFunction
#endif
#ifndef cudaGetErrorString
#  define cudaGetErrorString hipGetErrorString
#endif
#ifndef cudaGetErrorName
#  define cudaGetErrorName hipGetErrorName
#endif
#ifndef cudaGetLastError
#  define cudaGetLastError hipGetLastError
#endif
#ifndef cudaDeviceSynchronize
#  define cudaDeviceSynchronize hipDeviceSynchronize
#endif
#ifndef cuGetErrorName
#  define cuGetErrorName hipDrvGetErrorName
#endif
#ifndef cuDeviceGet
#  define cuDeviceGet hipDeviceGet
#endif
#ifndef CUdevice_attribute
#  define CUdevice_attribute hipDeviceAttribute_t
#endif
#ifndef CUjit_option
#  define CUjit_option hiprtcJIT_option
#endif
#ifndef CUlinkState
#  define CUlinkState hiprtcLinkState
#endif
#ifndef CUdeviceptr
#  define CUdeviceptr hipDeviceptr_t
#endif
#ifndef cuDeviCUfunction_attributeceGet
#  define CUfunction_attribute hipFunction_attribute
#endif
#ifndef CUoccupancyB2DSize
#  define CUoccupancyB2DSize void*
#endif
#ifndef CUstream
#  define CUstream hipStream_t
#endif
#ifndef CUjitInputType
#  define CUjitInputType hiprtcJITInputType
#endif
#ifndef CU_JIT_INPUT_NVVM
#  define CU_JIT_INPUT_NVVM HIPRTC_JIT_INPUT_LLVM_BITCODE
#endif
#ifndef CU_JIT_INPUT_PTX
#  define CU_JIT_INPUT_PTX HIPRTC_JIT_INPUT_CUBIN
#endif
#ifndef CU_JIT_INPUT_CUBIN
#  define CU_JIT_INPUT_CUBIN HIPRTC_JIT_INPUT_OBJECT
#endif
#ifndef CU_JIT_INPUT_FATBINARY
#  define CU_JIT_INPUT_FATBINARY HIPRTC_JIT_INPUT_FATBINARY
#endif
#ifndef CU_JIT_INPUT_OBJECT
#  define CU_JIT_INPUT_OBJECT HIPRTC_JIT_INPUT_OBJECT
#endif
#ifndef CU_JIT_INPUT_LIBRARY
#  define CU_JIT_INPUT_LIBRARY HIPRTC_JIT_INPUT_LIBRARY
#endif
#ifndef CU_JIT_INFO_LOG_BUFFER
#  define CU_JIT_INFO_LOG_BUFFER HIPRTC_JIT_INFO_LOG_BUFFER
#endif
#ifndef CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
#  define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES
#endif
#ifndef CU_JIT_ERROR_LOG_BUFFER
#  define CU_JIT_ERROR_LOG_BUFFER HIPRTC_JIT_ERROR_LOG_BUFFER
#endif
#ifndef CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
#  define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
#endif
#ifndef CUDA_ERROR_FILE_NOT_FOUND
#  define CUDA_ERROR_FILE_NOT_FOUND HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
#endif
#ifndef cudaDeviceProp
#  define cudaDeviceProp hipDeviceProp_t
#endif
#ifndef cudaGetDeviceCount
#  define cudaGetDeviceCount hipGetDeviceCount
#endif
#ifndef cuGetDeviceCount
#  define cuGetDeviceCount hipGetDeviceCount
#endif
#ifndef nvrtcVersion
#  define nvrtcVersion hiprtcVersion
#endif
#ifndef cudaMallocManaged
#  define cudaMallocManaged hipMallocManaged
#endif
#ifndef cudaFree
#  define cudaFree hipFree
#endif
#ifndef cudaMalloc
#  define cudaMalloc hipMalloc
#endif
#ifndef cudaMemcpyHostToDevice
#  define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#endif
#ifndef cudaMemcpyDeviceToHost
#  define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#endif
#ifndef cudaGetDevice
#  define cudaGetDevice hipGetDevice
#endif
#ifndef cudaDevAttrComputeCapabilityMajor
#  define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#endif
#ifndef cudaDevAttrComputeCapabilityMinor
#  define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#endif
#ifndef cudaMemcpy
#  define cudaMemcpy hipMemcpy
#endif
#ifndef cudaDeviceGetAttribute
#  define cudaDeviceGetAttribute hipDeviceGetAttribute
#endif
#ifndef cudaDeviceGetAttribute
#  define cudaDeviceGetAttribute hipDeviceGetAttribute
#endif
#ifndef CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#  define CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#endif
#ifndef CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
#  define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN hipDeviceAttributeSharedMemPerBlockOptin
#endif
#ifndef CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#  define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#endif
#ifndef CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#  define CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#endif
#ifndef cudaErrorAssert
#  define cudaErrorAssert hipErrorAssert
#endif
#ifndef cudaSetDevice
#  define cudaSetDevice hipSetDevice
#endif
#ifndef cuCtxGetDevice
#  define cuCtxGetDevice hipCtxGetDevice
#endif
#ifndef cuCtxGetCurrent
#  define cuCtxGetCurrent hipCtxGetCurrent
#endif
#ifndef cuDeviceGetAttribute
#  define cuDeviceGetAttribute hipDeviceGetAttribute
#endif
#ifndef cuGetDeviceProperties
#  define cuGetDeviceProperties hipGetDeviceProperties
#endif
#ifndef nvrtcAddNameExpression
#  define nvrtcAddNameExpression hiprtcAddNameExpression
#endif
#ifndef nvrtcLinkCreate
#  define nvrtcLinkCreate hiprtcLinkCreate
#endif
#ifndef cuDriverGetVersion
#  define cuDriverGetVersion hipDriverGetVersion
#endif
#ifndef cuModuleLoadData
#  define cuModuleLoadData hipModuleLoadData
#endif
#ifndef cuGetErrorString
#  define cuGetErrorString hipGetErrorString
#endif
#ifndef cuModuleUnload
#  define cuModuleUnload hipModuleUnload
#endif
#ifndef nvrtcLinkDestroy
#  define nvrtcLinkDestroy hiprtcLinkDestroy
#endif
#ifndef nvrtcLinkAddData
#  define nvrtcLinkAddData hiprtcLinkAddData
#endif
#ifndef nvrtcLinkComplete
#  define nvrtcLinkComplete hiprtcLinkComplete
#endif
#ifndef cuOccupancyMaxPotentialBlockSizeWithFlags
#  define cuOccupancyMaxPotentialBlockSizeWithFlags hipModuleOccupancyMaxPotentialBlockSizeWithFlags
#endif
#ifndef cuModuleGetGlobal
#  define cuModuleGetGlobal hipModuleGetGlobal
#endif
#ifndef cuFuncGetAttribute
#  define cuFuncGetAttribute hipFuncGetAttribute
#endif
#ifndef cuFuncSetAttribute
#  define cuFuncSetAttribute hipFuncSetAttribute
#endif
#ifndef cuMemcpyDtoHAsync
#  define cuMemcpyDtoHAsync hipMemcpyDtoHAsync
#endif
#ifndef cuMemcpyHtoDAsync
#  define cuMemcpyHtoDAsync hipMemcpyHtoDAsync
#endif
#ifndef nvrtcLinkAddFile
#  define nvrtcLinkAddFile hiprtcLinkAddFile
#endif
#ifndef CU_JIT_GENERATE_DEBUG_INFO
#  define CU_JIT_GENERATE_DEBUG_INFO HIPRTC_JIT_GENERATE_DEBUG_INFO
#endif
#ifndef CU_JIT_GENERATE_LINE_INFO
#  define CU_JIT_GENERATE_LINE_INFO HIPRTC_JIT_GENERATE_LINE_INFO
#endif
#ifndef CU_JIT_TARGET
#  define CU_JIT_TARGET HIPRTC_JIT_TARGET
#endif
#ifndef CU_JIT_OPTIMIZATION_LEVEL
#  define CU_JIT_OPTIMIZATION_LEVEL HIPRTC_JIT_OPTIMIZATION_LEVEL
#endif
#ifndef CU_JIT_LOG_VERBOSE
#  define CU_JIT_LOG_VERBOSE HIPRTC_JIT_LOG_VERBOSE
#endif
