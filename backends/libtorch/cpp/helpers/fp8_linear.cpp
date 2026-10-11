#include "helpers/fp8_linear.h"
#include <ATen/ops/_scaled_mm.h>
#include <string>
#include <mutex>
#include <thread>
#include <vector>
#include <tuple>
#include <cstdint>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/Context.h>
#include <ATen/cuda/Exceptions.h>
#include <cublasLt.h>
#include <cuda.h>
#include <c10/cuda/driver_api.h>
#endif

namespace tei {
/*
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

*/
// Descriptor construction follows PyTorch CUDA scaled_gemm (CUDABlas.cpp),
// retaining its BSD notice below. This cache changes host setup, not FP8 math.
struct Fp8RowwiseExecutor::Impl {
#if defined(TEI_TORCH_CUDA_KERNELS) && CUDA_VERSION >= 12090
  struct Plan {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t a{}, b{}, c{}, d{};
    cublasLtMatmulAlgo_t algorithm{};
    at::Tensor workspace;
    bool rejected=false;
    ~Plan() {
      if(d)cublasLtMatrixLayoutDestroy(d);
      if(c)cublasLtMatrixLayoutDestroy(c);
      if(b)cublasLtMatrixLayoutDestroy(b);
      if(a)cublasLtMatrixLayoutDestroy(a);
      if(operation)cublasLtMatmulDescDestroy(operation);
    }
  };
  struct Lane {
    int device;
    CUcontext context;
    std::thread::id thread;
    cudaStream_t stream;
    cublasLtHandle_t handle;
    at::Tensor workspace;
  };
  struct Entry {
    size_t lane, workspace_bytes;
    int64_t m,n,k;
    at::ScalarType output_dtype;
    std::unique_ptr<Plan> plan;
  };
  std::mutex mutex;
  std::vector<Lane> lanes;
  std::vector<Entry> entries;
  // Model-wide bounds: four execution lanes, 64 shapes/policies in total.
  // Capacity overflow uses ATen; existing plans/workspaces are never evicted.
  // Every lane allocates at most32MiB, shared by all its plans. Each plan
  // retains the allocation; changing workspace policy cannot free old storage.
  static constexpr size_t max_lanes=4, max_plans=64, max_workspace=32*1024*1024;
#endif
};
Fp8RowwiseExecutor::Fp8RowwiseExecutor():impl_(std::make_shared<Impl>()) {}
at::Tensor Fp8RowwiseExecutor::forward(const at::Tensor& x,const at::Tensor& weight,
  const at::Tensor& xs,const at::Tensor& ws,at::ScalarType output_dtype) const {
#if defined(TEI_TORCH_CUDA_KERNELS) && CUDA_VERSION >= 12090
  // Original learned Half and BF16 Wi/Wo packets prove separate output layouts.
  // Unknown output types retain ATen; model widths and real M are unchanged.
  if(output_dtype!=at::kHalf && output_dtype!=at::kBFloat16)return {};
  if(!x.is_cuda() || x.dim()!=2 || !weight.is_cuda() || weight.dim()!=2 ||
    x.scalar_type()!=at::kFloat8_e4m3fn || weight.scalar_type()!=at::kFloat8_e4m3fn ||
    x.device()!=weight.device() || x.size(0)<=0 || x.size(1)!=weight.size(1) ||
    !x.is_contiguous() || !weight.is_contiguous())return {};
  const auto m=x.size(0),n=weight.size(0),k=x.size(1);
  if(!((k==768 && n==2304)||(k==1152 && n==768)))return {};
  if(!xs.defined() || !ws.defined() || xs.device()!=x.device() || ws.device()!=x.device() ||
    xs.scalar_type()!=at::kFloat || ws.scalar_type()!=at::kFloat ||
    !xs.is_contiguous() || !ws.is_contiguous() ||
    xs.dim()!=2 || ws.dim()!=2 || xs.size(0)!=m || ws.size(0)!=n ||
    xs.size(1)!=1 || ws.size(1)!=1)return {};
  c10::cuda::CUDAGuard guard(x.device());
  const auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&capture));
  if(capture!=cudaStreamCaptureStatusNone)return {};
  if(at::globalContext()._SMCarveout_EXPERIMENTAL().has_value())return {};
  const auto* properties=at::cuda::getDeviceProperties(x.get_device());
  if(properties->major!=9 || properties->minor!=0)return {};
  // Reuse Torch's dynamically loaded driver API; no direct libcuda dependency.
  const auto get_context=c10::cuda::DriverAPI::get()->cuCtxGetCurrent_;
  CUcontext context{};
  if(!get_context || get_context(&context)!=CUDA_SUCCESS || !context)return {};
  static const auto version=cublasLtGetVersion();
  // The original learned-operand packet proves this library's first tactic.
  if(version!=130200)return {};
  const auto workspace_bytes=at::cuda::getCUDABlasLtWorkspaceSize();
  if(workspace_bytes>Impl::max_workspace)return {};
  const auto aligned=[](const at::Tensor& tensor) {
    return reinterpret_cast<uintptr_t>(tensor.const_data_ptr())%256==0;
  };
  if(!aligned(x)||!aligned(weight)||!aligned(xs)||!aligned(ws))return {};
  const auto handle=at::cuda::getCurrentCUDABlasLtHandle();
  const auto thread=std::this_thread::get_id();
  auto& cache=*impl_;
  std::lock_guard<std::mutex> lock(cache.mutex);
  size_t lane=0;
  for(;lane<cache.lanes.size();++lane) {
    const auto& v=cache.lanes[lane];
    if(v.device==x.get_device() && v.context==context && v.thread==thread &&
      v.stream==stream.stream() && v.handle==handle)break;
  }
  if(lane==cache.lanes.size()) {
    if(lane==Impl::max_lanes)return {};
    cache.lanes.push_back({x.get_device(),context,thread,stream.stream(),handle,{}});
  }
  Impl::Plan* plan=nullptr;
  for(auto& entry:cache.entries)if(entry.lane==lane && entry.workspace_bytes==workspace_bytes &&
    entry.m==m && entry.n==n && entry.k==k && entry.output_dtype==output_dtype) {plan=entry.plan.get();break;}
  if(!plan) {
    if(cache.entries.size()==Impl::max_plans)return {};
    auto owned=std::make_unique<Impl::Plan>();
    plan=owned.get();
    // Cache rejected geometry/policy too; no unbounded repeated host setup.
    cache.entries.push_back({lane,workspace_bytes,m,n,k,output_dtype,std::move(owned)});
    plan->rejected=true;
    const auto ok=[](cublasStatus_t status) { return status==CUBLAS_STATUS_SUCCESS; };
    if(!ok(cublasLtMatmulDescCreate(&plan->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F)))return {};
    const auto set=[&](cublasLtMatmulDescAttributes_t attr,const auto& value) {
      return ok(cublasLtMatmulDescSetAttribute(plan->operation,attr,&value,sizeof(value)));
    };
    const cublasOperation_t transpose=CUBLAS_OP_T,normal=CUBLAS_OP_N;
    const int8_t fast=1;
    const auto scale_mode=CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    const void* a_scale=ws.const_data_ptr(); const void* b_scale=xs.const_data_ptr();
    if(!set(CUBLASLT_MATMUL_DESC_TRANSA,transpose) || !set(CUBLASLT_MATMUL_DESC_TRANSB,normal) ||
      !set(CUBLASLT_MATMUL_DESC_FAST_ACCUM,fast) ||
      !set(CUBLASLT_MATMUL_DESC_A_SCALE_MODE,scale_mode) ||
      !set(CUBLASLT_MATMUL_DESC_B_SCALE_MODE,scale_mode) ||
      !set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,a_scale) ||
      !set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,b_scale))return {};
    const auto result_type=output_dtype==at::kHalf ? CUDA_R_16F : CUDA_R_16BF;
    if(!ok(cublasLtMatrixLayoutCreate(&plan->a,CUDA_R_8F_E4M3,k,n,k)) ||
      !ok(cublasLtMatrixLayoutCreate(&plan->b,CUDA_R_8F_E4M3,k,m,k)) ||
      !ok(cublasLtMatrixLayoutCreate(&plan->c,result_type,n,m,n)) ||
      !ok(cublasLtMatrixLayoutCreate(&plan->d,result_type,n,m,n)))return {};
    struct Preference {
      cublasLtMatmulPreference_t value{};
      ~Preference(){if(value)cublasLtMatmulPreferenceDestroy(value);}
    } preference;
    if(!ok(cublasLtMatmulPreferenceCreate(&preference.value)) ||
      !ok(cublasLtMatmulPreferenceSetAttribute(preference.value,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes))))return {};
    // Remaining preference defaults match Torch: alignment256, reductionmask7.
    cublasLtMatmulHeuristicResult_t heuristic{}; int returned=0;
    if(!ok(cublasLtMatmulAlgoGetHeuristic(handle,plan->operation,plan->a,plan->b,
      plan->c,plan->d,preference.value,1,&heuristic,&returned)) || returned!=1 ||
      heuristic.state!=CUBLAS_STATUS_SUCCESS || heuristic.workspaceSize>workspace_bytes)return {};
    int id=0;size_t written=0;
    if(!ok(cublasLtMatmulAlgoConfigGetAttribute(&heuristic.algo,CUBLASLT_ALGO_CONFIG_ID,
      &id,sizeof(id),&written)) || written!=sizeof(id) || id!=66)return {};
    plan->algorithm=heuristic.algo; // Original first tactic, not a custom search.
    auto& workspace=cache.lanes[lane].workspace;
    if(workspace_bytes && !workspace.defined())
      workspace=at::empty({static_cast<int64_t>(Impl::max_workspace)},x.options().dtype(at::kByte));
    plan->workspace=workspace; // Keep this exact allocation alive independently.
    plan->rejected=false;
  }
  if(plan->rejected)return {};
  const void* a_scale=ws.const_data_ptr(); const void* b_scale=xs.const_data_ptr();
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(plan->operation,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&a_scale,sizeof(a_scale)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(plan->operation,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&b_scale,sizeof(b_scale)));
  auto output=at::empty({m,n},x.options().dtype(output_dtype));
  if(!aligned(output))return {};
  if(plan->workspace.defined())c10::cuda::CUDACachingAllocator::recordStream(
    plan->workspace.storage().data_ptr(),stream);
  const float alpha=1.f,beta=0.f;
  const auto status=cublasLtMatmul(handle,plan->operation,&alpha,
    weight.const_data_ptr(),plan->a,x.const_data_ptr(),plan->b,&beta,
    weight.const_data_ptr(),plan->c,output.mutable_data_ptr(),plan->d,
    &plan->algorithm,workspace_bytes?plan->workspace.mutable_data_ptr():nullptr,
    workspace_bytes,stream.stream());
  if(status==CUBLAS_STATUS_NOT_SUPPORTED) { plan->rejected=true;return {}; }
  TORCH_CUDABLAS_CHECK(status);
  return output;
#else
  return {};
#endif
}

// Ordinary Modern projections: no quantization or scale descriptors.
struct OrdinaryLinearExecutor::Impl {
#if defined(TEI_TORCH_CUDA_KERNELS) && CUDA_VERSION >= 12090
  struct Plan {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t a{}, b{}, c{}, d{};
    cublasLtMatmulAlgo_t algorithm{};
    at::Tensor workspace;
    bool rejected=false;
    ~Plan() {
      if(d)cublasLtMatrixLayoutDestroy(d);
      if(c)cublasLtMatrixLayoutDestroy(c);
      if(b)cublasLtMatrixLayoutDestroy(b);
      if(a)cublasLtMatrixLayoutDestroy(a);
      if(operation)cublasLtMatmulDescDestroy(operation);
    }
  };
  struct Lane {
    int device;
    CUcontext context;
    std::thread::id thread;
    cudaStream_t stream;
    cublasLtHandle_t handle;
    at::Tensor workspace;
  };
  struct Entry {
    size_t lane, workspace_bytes;
    int64_t m,n,k;
    at::ScalarType output_dtype;
    std::unique_ptr<Plan> plan;
  };
  std::mutex mutex;
  std::vector<Lane> lanes;
  std::vector<Entry> entries;
  // Model-wide bounds: four execution lanes, 64 shapes/policies in total.
  // Capacity overflow uses ATen; existing plans/workspaces are never evicted.
  // Every lane allocates at most32MiB, shared by all its plans. Each plan
  // retains the allocation; changing workspace policy cannot free old storage.
  static constexpr size_t max_lanes=4, max_plans=64, max_workspace=32*1024*1024;
#endif
};
OrdinaryLinearExecutor::OrdinaryLinearExecutor():impl_(std::make_shared<Impl>()) {}
at::Tensor OrdinaryLinearExecutor::forward(const at::Tensor& x,const at::Tensor& weight) const {
#if defined(TEI_TORCH_CUDA_KERNELS) && CUDA_VERSION >= 12090
  // Independent saved Half/BF16 ordinary packets define these layouts.
  // Unknown dtype/strides/policies retain original ATen.
  if(!x.is_cuda() || x.dim()!=2 || !weight.is_cuda() || weight.dim()!=2 ||
    (x.scalar_type()!=at::kHalf && x.scalar_type()!=at::kBFloat16) ||
    weight.scalar_type()!=x.scalar_type() ||
    x.device()!=weight.device() || x.size(0)<=0 || x.size(1)!=weight.size(1) ||
    !x.is_contiguous() || !weight.is_contiguous())return {};
  const auto m=x.size(0),n=weight.size(0),k=x.size(1);
  if(!((k==768 && (n==2304 || n==768))||(k==1152 && n==768)))return {};
  const auto output_dtype=x.scalar_type();
  c10::cuda::CUDAGuard guard(x.device());
  const auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&capture));
  if(capture!=cudaStreamCaptureStatusNone)return {};
  if(at::cuda::tunable::getTuningContext()->IsTunableOpEnabled() ||
    at::globalContext()._SMCarveout_EXPERIMENTAL().has_value() ||
    (output_dtype==at::kHalf && at::globalContext().allowFP16AccumulationCuBLAS()) ||
    (output_dtype==at::kHalf ? at::globalContext().allowFP16ReductionCuBLAS() :
      at::globalContext().allowBF16ReductionCuBLAS())!=
      at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK ||
    at::globalContext().deterministicAlgorithms() ||
    at::globalContext().blasPreferredBackend()!=at::BlasBackend::Cublaslt)return {};
  const auto* properties=at::cuda::getDeviceProperties(x.get_device());
  if(properties->major!=9 || properties->minor!=0)return {};
  // Reuse Torch's dynamically loaded driver API; no direct libcuda dependency.
  const auto get_context=c10::cuda::DriverAPI::get()->cuCtxGetCurrent_;
  CUcontext context{};
  if(!get_context || get_context(&context)!=CUDA_SUCCESS || !context)return {};
  static const auto version=cublasLtGetVersion();
  // The original learned-operand packet proves this library's first tactic.
  if(version!=130200)return {};
  const auto workspace_bytes=at::cuda::getCUDABlasLtWorkspaceSize();
  if(workspace_bytes>Impl::max_workspace)return {};
  const auto aligned=[](const at::Tensor& tensor) {
    return reinterpret_cast<uintptr_t>(tensor.const_data_ptr())%256==0;
  };
  if(!aligned(x)||!aligned(weight))return {};
  const auto handle=at::cuda::getCurrentCUDABlasLtHandle();
  const auto thread=std::this_thread::get_id();
  auto& cache=*impl_;
  std::lock_guard<std::mutex> lock(cache.mutex);
  size_t lane=0;
  for(;lane<cache.lanes.size();++lane) {
    const auto& v=cache.lanes[lane];
    if(v.device==x.get_device() && v.context==context && v.thread==thread &&
      v.stream==stream.stream() && v.handle==handle)break;
  }
  if(lane==cache.lanes.size()) {
    if(lane==Impl::max_lanes)return {};
    cache.lanes.push_back({x.get_device(),context,thread,stream.stream(),handle,{}});
  }
  Impl::Plan* plan=nullptr;
  for(auto& entry:cache.entries)if(entry.lane==lane && entry.workspace_bytes==workspace_bytes &&
    entry.m==m && entry.n==n && entry.k==k && entry.output_dtype==output_dtype) {plan=entry.plan.get();break;}
  if(!plan) {
    if(cache.entries.size()==Impl::max_plans)return {};
    auto owned=std::make_unique<Impl::Plan>();
    plan=owned.get();
    // Cache rejected geometry/policy too; no unbounded repeated host setup.
    cache.entries.push_back({lane,workspace_bytes,m,n,k,output_dtype,std::move(owned)});
    plan->rejected=true;
    const auto ok=[](cublasStatus_t status) { return status==CUBLAS_STATUS_SUCCESS; };
    if(!ok(cublasLtMatmulDescCreate(&plan->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F)))return {};
    const auto set=[&](cublasLtMatmulDescAttributes_t attr,const auto& value) {
      return ok(cublasLtMatmulDescSetAttribute(plan->operation,attr,&value,sizeof(value)));
    };
    const cublasOperation_t transpose=CUBLAS_OP_T,normal=CUBLAS_OP_N;
    const int8_t fast=0;
    const cublasLtEpilogue_t epilogue=CUBLASLT_EPILOGUE_DEFAULT;
    if(!set(CUBLASLT_MATMUL_DESC_TRANSA,transpose) || !set(CUBLASLT_MATMUL_DESC_TRANSB,normal) ||
      !set(CUBLASLT_MATMUL_DESC_FAST_ACCUM,fast) ||
      !set(CUBLASLT_MATMUL_DESC_EPILOGUE,epilogue))return {};
    const auto type=output_dtype==at::kHalf ? CUDA_R_16F : CUDA_R_16BF;
    if(!ok(cublasLtMatrixLayoutCreate(&plan->a,type,k,n,k)) ||
      !ok(cublasLtMatrixLayoutCreate(&plan->b,type,k,m,k)) ||
      !ok(cublasLtMatrixLayoutCreate(&plan->c,type,n,m,n)) ||
      !ok(cublasLtMatrixLayoutCreate(&plan->d,type,n,m,n)))return {};
    struct Preference {
      cublasLtMatmulPreference_t value{};
      ~Preference(){if(value)cublasLtMatmulPreferenceDestroy(value);}
    } preference;
    if(!ok(cublasLtMatmulPreferenceCreate(&preference.value)) ||
      !ok(cublasLtMatmulPreferenceSetAttribute(preference.value,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes))))return {};
    // Remaining preference defaults match Torch: alignment256, reductionmask7.
    cublasLtMatmulHeuristicResult_t heuristic{}; int returned=0;
    if(!ok(cublasLtMatmulAlgoGetHeuristic(handle,plan->operation,plan->a,plan->b,
      plan->c,plan->d,preference.value,1,&heuristic,&returned)) || returned!=1 ||
      heuristic.state!=CUBLAS_STATUS_SUCCESS || heuristic.workspaceSize>workspace_bytes)return {};
    int id=0;size_t written=0;
    if(!ok(cublasLtMatmulAlgoConfigGetAttribute(&heuristic.algo,CUBLASLT_ALGO_CONFIG_ID,
      &id,sizeof(id),&written)) || written!=sizeof(id) || id!=66)return {};
    plan->algorithm=heuristic.algo; // Original first tactic, not a custom search.
    auto& workspace=cache.lanes[lane].workspace;
    if(workspace_bytes && !workspace.defined())
      workspace=at::empty({static_cast<int64_t>(Impl::max_workspace)},x.options().dtype(at::kByte));
    plan->workspace=workspace; // Keep this exact allocation alive independently.
    plan->rejected=false;
  }
  if(plan->rejected)return {};
  auto output=at::empty({m,n},x.options().dtype(output_dtype));
  if(!aligned(output))return {};
  if(plan->workspace.defined())c10::cuda::CUDACachingAllocator::recordStream(
    plan->workspace.storage().data_ptr(),stream);
  const float alpha=1.f,beta=0.f;
  const auto status=cublasLtMatmul(handle,plan->operation,&alpha,
    weight.const_data_ptr(),plan->a,x.const_data_ptr(),plan->b,&beta,
    output.const_data_ptr(),plan->c,output.mutable_data_ptr(),plan->d,
    &plan->algorithm,workspace_bytes?plan->workspace.mutable_data_ptr():nullptr,
    workspace_bytes,stream.stream());
  if(status==CUBLAS_STATUS_NOT_SUPPORTED) { plan->rejected=true;return {}; }
  TORCH_CUDABLAS_CHECK(status);
  return output;
#else
  return {};
#endif
}

#ifndef TEI_TORCH_CUDA_KERNELS
std::pair<at::Tensor, at::Tensor> fp8_quantize_rows(const at::Tensor&) {
  TORCH_CHECK(false, "Dynamic FP8 requires a CUDA LibTorch build");
}
std::pair<at::Tensor,at::Tensor> fp8_quantize_gated_rows(const at::Tensor&) {
  TORCH_CHECK(false,"Gated dynamic FP8 requires a CUDA LibTorch build");
}
#endif
Fp8Linear::Fp8Linear(const at::Tensor& weight, bool cached_rowwise,
  std::shared_ptr<Fp8RowwiseExecutor> executor, bool gated_quantization)
  : dtype_(weight.scalar_type()), rowwise_(false), gated_quantization_(gated_quantization),
    cached_executor_(cached_rowwise ? (executor ? std::move(executor) :
      std::make_shared<Fp8RowwiseExecutor>()) : nullptr) {
  TORCH_CHECK(weight.is_cuda() && weight.dim()==2 && weight.size(0)>0 && weight.size(1)>0,
    "Dynamic FP8 requires nonempty CUDA matrix weights");
  TORCH_CHECK(dtype_==at::kHalf || dtype_==at::kBFloat16,
    "Dynamic FP8 requires an unquantized float16/bfloat16 checkpoint");
  TORCH_CHECK(weight.size(0)%16==0 && weight.size(1)%16==0,
    "Dynamic FP8 projection widths must be divisible by16; tokens are never padded");
#ifdef TEI_TORCH_CUDA_KERNELS
  c10::cuda::CUDAGuard guard(weight.device());
  const auto* props=at::cuda::getDeviceProperties(weight.device().index());
  const int capability=props->major*10+props->minor;
  TORCH_CHECK(capability==89 || capability==90 || capability==100 || capability==120,
    "Dynamic FP8 requires SM89/90/100/120");
  rowwise_=capability==90;
#else
  TORCH_CHECK(false, "Dynamic FP8 requires a CUDA LibTorch build");
#endif
  TORCH_CHECK(at::isfinite(weight).all().item<bool>(), "Dynamic FP8 checkpoint weights must be finite");
  auto quantized=fp8_quantize_rows(weight.contiguous());
  matrix_=std::move(quantized.first);scales_=std::move(quantized.second);
  unit_=at::ones({},weight.options().dtype(at::kFloat));
}
at::Tensor Fp8Linear::forward(const at::Tensor& input,const at::Tensor& bias) const {
  TORCH_CHECK(input.dim()==2 && input.device()==matrix_.device()
    && input.scalar_type()==dtype_ && input.size(1)==input_width(),
    "Invalid dynamic FP8 packed activations");
  TORCH_CHECK(!bias.defined() || (bias.dim()==1 && bias.size(0)==output_width()
    && bias.device()==input.device() && bias.scalar_type()==dtype_), "Invalid dynamic FP8 bias");
  if(input.size(0)==0)return at::empty({0,output_width()},input.options());
#ifdef TEI_TORCH_CUDA_KERNELS
  c10::cuda::CUDAGuard guard(input.device());
#endif
  auto quantized=fp8_quantize_rows(input.contiguous());
  return apply_quantized(quantized.first,quantized.second,bias);
}
at::Tensor Fp8Linear::forward_quantized(const at::Tensor& input,const at::Tensor& scales,
  const at::Tensor& bias) const {
  TORCH_CHECK(input.dim()==2 && input.device()==matrix_.device() && input.is_contiguous()
    && input.scalar_type()==at::kFloat8_e4m3fn && input.size(1)==input_width(),
    "Invalid prequantized FP8 packed activations");
  TORCH_CHECK(scales.device()==input.device() && scales.scalar_type()==at::kFloat
    && scales.is_contiguous() && scales.dim()==2 && scales.size(0)==input.size(0) && scales.size(1)==1,
    "Invalid prequantized FP8 row scales");
  TORCH_CHECK(!bias.defined() || (bias.dim()==1 && bias.size(0)==output_width()
    && bias.device()==input.device() && bias.scalar_type()==dtype_),"Invalid dynamic FP8 bias");
  if(input.size(0)==0)return at::empty({0,output_width()},input.options().dtype(dtype_));
#ifdef TEI_TORCH_CUDA_KERNELS
  c10::cuda::CUDAGuard guard(input.device());
#endif
  return apply_quantized(input,scales,bias);
}
at::Tensor Fp8Linear::apply_quantized(const at::Tensor& input,const at::Tensor& scales,
  const at::Tensor& bias) const {
  at::Tensor output;
  if(rowwise_) {
    if(cached_executor_)output=cached_executor_->forward(input,matrix_,
      scales,scales_,dtype_);
    if(!output.defined())try {
      // Candle's Hopper recipe prefers fast accumulation. Some CUDA versions
      // lack fast tactics for short, wide-K shapes; preserve its full fallback.
      output=at::_scaled_mm(input,matrix_.t(),scales,
        scales_.t(),{}, {},dtype_,true);
    } catch(const c10::Error& error) {
      if(std::string(error.what_without_backtrace()).find("CUBLAS_STATUS_NOT_SUPPORTED")==std::string::npos)throw;
      output=at::_scaled_mm(input,matrix_.t(),scales,
        scales_.t(),{}, {},dtype_,false);
    }
  } else {
    // Match the portable recipe: scale full FP32 accumulation before the
    // single model-dtype rounding. No tensorwide activation scale is inferred.
    output=(at::_scaled_mm(input,matrix_.t(),unit_,unit_,{}, {},at::kFloat,false)
      *scales*scales_.t()).to(dtype_);
  }
  // Candle's quantized MLP applies bias after GEMM model-dtype rounding.
  return bias.defined()?output+bias:output;
}
}
