# [copy] Route converting copies through the vectorized elementwise kernel

Fixes #6150

## Summary

A converting copy that is not a plain float cast (int32 -> int64,
bool -> int64, bool -> float64) launches `UnrolledElementwiseKernel` with
`LoadWithCast`/`StoreWithCast` on XPU, while CUDA dispatches the same copies
to `gpu_kernel_nocast` with a converting functor, landing in
`vectorized_elementwise_kernel`
(`aten/src/ATen/native/cuda/Copy.cu`, `direct_copy_kernel_cuda` /
`converting_copy_kernel_cuda`). The copy result is correct on XPU, only the
kernel choice is wrong, so `test_converting_copy_emits_vectorized_kernel`
(which asserts the vectorized kernel name appears in the profile) fails on
XPU.

## Root cause

`copy_kernel` (`src/ATen/native/xpu/sycl/CopyKernel.cpp`) only routed the
float16/bfloat16 <-> float pairs to `gpu_kernel_nocast` (the
`VectorizedElementwiseKernel` path). Every other converting dtype pair fell
into the final dtype-dispatched `gpu_kernel(iter, CopyScalarFunc<scalar_t>())`
arm, which launches `UnrolledElementwiseKernel` with dynamic-cast loads.

## Fix (mirrors cuda/Copy.cu)

Add the same three explicit converting pairs CUDA enumerates, right before
the generic fallback:

```cpp
} else if (iter.dtype(1) == kInt && iter.dtype(0) == kLong) {
  gpu_kernel_nocast(iter, CastScalarFunc<int32_t, int64_t>());
} else if (iter.dtype(1) == kBool && iter.dtype(0) == kLong) {
  gpu_kernel_nocast(iter, CastScalarFunc<bool, int64_t>());
} else if (iter.dtype(1) == kBool && iter.dtype(0) == kDouble) {
  gpu_kernel_nocast(iter, CastScalarFunc<bool, double>());
}
```

`gpu_kernel_nocast` + `CastScalarFunc` is the exact XPU equivalent of the
CUDA `converting_copy_kernel_cuda` helper (functor converts, no dynamic
casting), and is already used by the float16/bfloat16 <-> float path in the
same file.

## Verification (Arc B580, incremental build)

Kineto device profiling is not available on the local box (profiler XPU
factory registration issue, see #6105), so dispatch was verified with
temporary `TORCH_WARN` markers in the new arms (removed from this branch):

* int32->int64, bool->int64, bool->float64: all three hit the new
  `gpu_kernel_nocast` path (marker fired, i.e. `VectorizedElementwiseKernel`).
* Control copies that must NOT change (int64->int32, bf16->fp32): no marker,
  unchanged dispatch.
* Correctness: `dst.cpu() == src.cpu().to(dtype)` for all three pairs.
* Perf (8192x8192, 50 reps): bool->int64 1.16 -> 0.76 ms, bool->float64
  1.25 -> 0.81 ms (~35% faster); int32->int64 flat (bandwidth-bound,
  vec width capped at 2 by the 8-byte store).
* `pytest test_torch.py -k "(copy or _to_) and xpu"`: 94 passed /
  93 pre-existing skips / 0 failures.

With this, the XPU variant of `test_converting_copy_emits_vectorized_kernel`
can be enabled alongside the CUDA one once the test enablement lands upstream.
