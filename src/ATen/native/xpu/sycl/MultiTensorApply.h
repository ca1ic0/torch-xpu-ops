/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
#include <ATen/ceil_div.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/ScalarType.h>
#include <comm/xpu_aten.h>
#include <vector>

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/native/xpu/sycl/GroupReduceUtils.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

// Instruction level Parallelism, namely vec size
static constexpr int64_t kILP = 4;
template <typename T>
bool is_aligned(T* p) {
  return ((uint64_t)p) % (kILP * sizeof(T)) == 0;
}

template <typename T, int64_t kILP = kILP>
void load_store(T* dst, T* src, int dst_offset, int src_offset) {
  using LT = at::native::memory::aligned_vector<T, kILP>;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template <typename scalar_vals_t, int n>
struct TLMetaForAddressScalar {
  void* addresses[n];
  uint32_t numel_to_tensor;
  scalar_vals_t scalar_vals;
};

template <int n>
struct TLMetaForAddress {
  void* addresses[n];
  uint32_t numel_to_tensor;
};

template <int n>
struct TLFusedMetaForAddress {
  void* addresses[n];
  uint32_t numel_to_tensor;
  void* state_steps_addresses;
};

struct TLMetaForWG {
  uint32_t wg_to_tensor;
  uint32_t wg_to_chunk;
};

static int64_t multi_tensor_apply_kernel_get_wg_size(int simd) {
  return get_group_reduce_group_size(simd);
}

static int64_t multi_tensor_apply_kernel_get_chunk_size(
    int simd,
    size_t total_elements) {
  int64_t wg_size = at::xpu::getDeviceMaxWorkGroupSize();
  int64_t sub_groups_per_wg = wg_size / simd;
  int64_t hw_threads = at::xpu::getDeviceHWThreads();
  int64_t num_wgs = at::ceil_div(hw_threads, sub_groups_per_wg);
  int64_t element_per_thread =
      (int64_t)total_elements / (num_wgs * wg_size);
  element_per_thread = std::max(element_per_thread, (int64_t)1);
  int64_t max_wg_size = multi_tensor_apply_kernel_get_wg_size(simd);
  return max_wg_size * element_per_thread;
}

static inline int64_t multi_tensor_apply_fused_kernel_get_wg_size() {
  return syclMaxWorkItemsPerSubSlice();
}

static inline int64_t multi_tensor_apply_fused_kernel_get_chunk_size() {
  int64_t max_wg_size = multi_tensor_apply_fused_kernel_get_wg_size();
  return max_wg_size * kILP;
}

template <typename T, typename Y, typename U, typename... ArgTypes>
struct MultiTensorApplyKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item_id) const {
    // Expand the tuple elements manually and call the callable
    expandAndCall(item_id, std::index_sequence_for<ArgTypes...>());
  }
  MultiTensorApplyKernelFunctor(
      int64_t chunk_size_,
      T tlAddressMeta_,
      Y tlWGMeta_,
      U callable_,
      ArgTypes... args_)
      : chunk_size(chunk_size_),
        tlAddressMeta(tlAddressMeta_),
        tlWGMeta(tlWGMeta_),
        callable(callable_),
        args(std::make_tuple(args_...)) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    if constexpr (std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, U>) {
      callable.sycl_ker_config_convention(cgh);
    }
  }

 private:
  template <std::size_t... Indices>
  void expandAndCall(sycl::nd_item<1> item_id, std::index_sequence<Indices...>)
      const {
    // Call the callable with expanded tuple elements
    callable(
        chunk_size,
        tlAddressMeta,
        tlWGMeta,
        item_id,
        std::get<Indices>(args)...);
  }

  int64_t chunk_size;
  T tlAddressMeta;
  Y tlWGMeta;
  U callable;
  std::tuple<ArgTypes...> args;
};

template <
    bool fused_kernel,
    typename T,
    typename Y,
    typename U,
    typename... ArgTypes>
void launch_multi_tensor_apply_kernel(
    T tlAddressMeta,
    Y tlWGMeta,
    U callable,
    int num_wg,
    int64_t chunk_size,
    ArgTypes... args) {
  auto& q = getCurrentSYCLQueue();
  int64_t simd = at::xpu::getDeviceMaxSubGroupSize();
  int64_t max_wg_size = multi_tensor_apply_kernel_get_wg_size(simd);

  if constexpr (fused_kernel) {
    max_wg_size = multi_tensor_apply_fused_kernel_get_wg_size();
    chunk_size = multi_tensor_apply_fused_kernel_get_chunk_size();
  }

  MultiTensorApplyKernelFunctor<T, Y, U, ArgTypes...> kfn(
      chunk_size, tlAddressMeta, tlWGMeta, callable, args...);

  sycl_kernel_submit(
      sycl::range<1>(num_wg * max_wg_size),
      sycl::range<1>(max_wg_size),
      q,
      kfn);
}

template <int depth, typename scalar_t, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::ArrayRef<Scalar> scalars,
    T callable,
    ArgTypes... args) {
  using scalar_vals_t = typename T::opmath_t;
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match he depth");
  size_t n_tensors = tensor_lists[0].size();

  auto& q = getCurrentSYCLQueue();
  int simd = at::xpu::getDeviceMaxSubGroupSize();
  size_t total_elements = 0;
  for (size_t t = 0; t < n_tensors; ++t)
    total_elements += tensor_lists[0][t].numel();
  int64_t chunk_size =
      multi_tensor_apply_kernel_get_chunk_size(simd, total_elements);
  uint64_t totalWG = 0;
  for (size_t t = 0; t < n_tensors; ++t)
    totalWG += at::ceil_div(tensor_lists[0][t].numel(), chunk_size);

  const size_t addr_bytes =
      sizeof(TLMetaForAddressScalar<scalar_vals_t, depth>) * n_tensors;
  const size_t wg_bytes = sizeof(TLMetaForWG) * totalWG;
  auto meta_dptr = c10::GetAllocator(c10::DeviceType::XPU)->allocate(
      addr_bytes + wg_bytes);
  auto metaAddressInput = static_cast<
      TLMetaForAddressScalar<scalar_vals_t, depth>*>(meta_dptr.get());
  auto metaWGInput = reinterpret_cast<TLMetaForWG*>(
      static_cast<char*>(meta_dptr.get()) + addr_bytes);
  auto hst_dptr =
      at::getHostAllocator(at::kXPU)->allocate(addr_bytes + wg_bytes);
  auto tlAddress = static_cast<
      TLMetaForAddressScalar<scalar_vals_t, depth>*>(hst_dptr.get());
  auto tlWGMeta = reinterpret_cast<TLMetaForWG*>(
      static_cast<char*>(hst_dptr.get()) + addr_bytes);

  // this loop record all the tensor address and numel info.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    tlAddress[t].numel_to_tensor = numel;
    tlAddress[t].scalar_vals = scalars[t].to<scalar_t>();
    for (int d = 0; d < depth; ++d) {
      tlAddress[t].addresses[d] = tensor_lists[d][t].mutable_data_ptr();
    }
  }
  uint64_t posWG = 0;
  // this loop record the correspond tensor and chunk info for each work group.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    auto chunkForWG = (numel + chunk_size - 1) / chunk_size;
    for (size_t chunkId = 0; chunkId < chunkForWG; ++chunkId, ++posWG) {
      tlWGMeta[posWG].wg_to_tensor = t;
      tlWGMeta[posWG].wg_to_chunk = chunkId;
    }
  }
  TORCH_CHECK(
      posWG == totalWG,
      "Work group index dose not equal to the allocated memory size, segment fault might occur");

  q.memcpy(meta_dptr.get(), hst_dptr.get(), addr_bytes + wg_bytes);
  at::getHostAllocator(at::kXPU)->record_event(
      static_cast<void*>(tlAddress),
      hst_dptr.get_context(),
      at::xpu::getCurrentXPUStream());

  launch_multi_tensor_apply_kernel<false>(
      metaAddressInput, metaWGInput, callable, totalWG, chunk_size, args...);
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match he depth");
  size_t n_tensors = tensor_lists[0].size();

  auto& q = getCurrentSYCLQueue();
  int simd = at::xpu::getDeviceMaxSubGroupSize();
  size_t total_elements = 0;
  for (size_t t = 0; t < n_tensors; ++t)
    total_elements += tensor_lists[0][t].numel();
  int64_t chunk_size =
      multi_tensor_apply_kernel_get_chunk_size(simd, total_elements);
  uint64_t totalWG = 0;
  for (size_t t = 0; t < n_tensors; ++t)
    totalWG += at::ceil_div(tensor_lists[0][t].numel(), chunk_size);

  const size_t addr_bytes = sizeof(TLMetaForAddress<depth>) * n_tensors;
  const size_t wg_bytes = sizeof(TLMetaForWG) * totalWG;
  auto meta_dptr = c10::GetAllocator(c10::DeviceType::XPU)->allocate(
      addr_bytes + wg_bytes);
  auto metaAddressInput =
      static_cast<TLMetaForAddress<depth>*>(meta_dptr.get());
  auto metaWGInput = reinterpret_cast<TLMetaForWG*>(
      static_cast<char*>(meta_dptr.get()) + addr_bytes);
  auto hst_dptr =
      at::getHostAllocator(at::kXPU)->allocate(addr_bytes + wg_bytes);
  auto tlAddress =
      static_cast<TLMetaForAddress<depth>*>(hst_dptr.get());
  auto tlWGMeta = reinterpret_cast<TLMetaForWG*>(
      static_cast<char*>(hst_dptr.get()) + addr_bytes);

  // this loop record all the tensor address and numel info.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    tlAddress[t].numel_to_tensor = numel;
    for (int d = 0; d < depth; ++d) {
      tlAddress[t].addresses[d] = tensor_lists[d][t].mutable_data_ptr();
    }
  }
  uint64_t posWG = 0;
  // this loop record the correspond tensor and chunk info for each work group.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    auto chunkForWG = (numel + chunk_size - 1) / chunk_size;
    for (size_t chunkId = 0; chunkId < chunkForWG; ++chunkId, ++posWG) {
      tlWGMeta[posWG].wg_to_tensor = t;
      tlWGMeta[posWG].wg_to_chunk = chunkId;
    }
  }
  TORCH_CHECK(
      posWG == totalWG,
      "Work group index dose not equal to the allocated memory size, segment fault might occur");

  q.memcpy(meta_dptr.get(), hst_dptr.get(), addr_bytes + wg_bytes);
  at::getHostAllocator(at::kXPU)->record_event(
      static_cast<void*>(tlAddress),
      hst_dptr.get_context(),
      at::xpu::getCurrentXPUStream());

  launch_multi_tensor_apply_kernel<false>(
      metaAddressInput, metaWGInput, callable, totalWG, chunk_size, args...);
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply_for_fused_optimizer(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::TensorList state_steps,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth");
  const auto n_tensors = tensor_lists[0].size();

  auto& q = getCurrentSYCLQueue();
  int64_t chunk_size = multi_tensor_apply_fused_kernel_get_chunk_size();
  uint64_t totalWG = 0;
  for (size_t t = 0; t < n_tensors; ++t)
    totalWG += at::ceil_div(tensor_lists[0][t].numel(), chunk_size);

  const size_t addr_bytes = sizeof(TLFusedMetaForAddress<depth>) * n_tensors;
  const size_t wg_bytes = sizeof(TLMetaForWG) * totalWG;
  auto meta_dptr = c10::GetAllocator(c10::DeviceType::XPU)->allocate(
      addr_bytes + wg_bytes);
  auto metaFusedAddressInput = static_cast<
      TLFusedMetaForAddress<depth>*>(meta_dptr.get());
  auto metaWGInput = reinterpret_cast<TLMetaForWG*>(
      static_cast<char*>(meta_dptr.get()) + addr_bytes);
  auto hst_dptr =
      at::getHostAllocator(at::kXPU)->allocate(addr_bytes + wg_bytes);
  auto tlAddress = static_cast<
      TLFusedMetaForAddress<depth>*>(hst_dptr.get());
  auto tlWGMeta = reinterpret_cast<TLMetaForWG*>(
      static_cast<char*>(hst_dptr.get()) + addr_bytes);

  // this loop record all the tensor address and numel info.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    tlAddress[t].numel_to_tensor = numel;
    tlAddress[t].state_steps_addresses = state_steps[t].mutable_data_ptr();
    for (int d = 0; d < depth; ++d) {
      tlAddress[t].addresses[d] = tensor_lists[d][t].mutable_data_ptr();
    }
  }
  uint64_t posWG = 0;
  // this loop record the correspond tensor and chunk info for each work group.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    auto chunkForWG = (numel + chunk_size - 1) / chunk_size;
    for (size_t chunkId = 0; chunkId < chunkForWG; ++chunkId, ++posWG) {
      tlWGMeta[posWG].wg_to_tensor = t;
      tlWGMeta[posWG].wg_to_chunk = chunkId;
    }
  }
  TORCH_CHECK(
      posWG == totalWG,
      "Work group index dose not equal to the allocated memory size, segment fault might occur");

  q.memcpy(meta_dptr.get(), hst_dptr.get(), addr_bytes + wg_bytes);
  at::getHostAllocator(at::kXPU)->record_event(
      static_cast<void*>(tlAddress),
      hst_dptr.get_context(),
      at::xpu::getCurrentXPUStream());

  launch_multi_tensor_apply_kernel<true>(
      metaFusedAddressInput, metaWGInput, callable, totalWG, chunk_size,
      args...);
}

} // namespace at::native::xpu
