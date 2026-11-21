# [XPU] Only take the split_with_sizes_copy.out fast path for dense, matching slices

Fixes #6321

## Summary

Port of pytorch#203892 (commit `b0c56ca7`, "[CUDA] Only take the
split_with_sizes_copy.out fast path for dense, matching slices") to the XPU
kernel.

`torch.split_with_sizes_copy(..., out=...)` on XPU took its fused fast-path
kernel for any `self.is_non_overlapping_and_dense()` input and used
`stride(dim)` as the per-split chunk size / split base offset. For dense but
non-canonically-laid-out operands (transposed / channels-last / as_strided
views) `stride(dim)` differs from the trailing-numel product, so the kernel

* computes split base offsets and chunk sizes from the wrong multiplier,
  returning wrong values (e.g. a transposed 3x4 input split along dim 0
  returns `[0, 1, 0, 0, 0, 0]` instead of `[0, 4, 8, 1, 5, 9]`), and
* copies `split_size * stride(dim) * elem_size` bytes per chunk, writing past
  the end of `out` tensors whose memory is smaller than the (dense) slice
  (out-of-bounds writes).

CUDA fixed this by gating the fast path on `has_dense_matching_slices` and
computing offsets/sizes from `slice_numel`. This PR ports that exactly:

* `get_split_base_addrs` / `get_split_chunk_sizes` use
  `c10::multiply_integers(sizes.slice(dim + 1))` instead of `strides[dim]`.
* New `has_dense_order` / `has_dense_matching_slices` gates: the fast path is
  only taken when every (non-empty) output is dense with the same dim order as
  `self` and `self.stride(dim)` equals the trailing-numel product (or there is
  only a single non-empty split, which is a flat copy). Everything else falls
  back to the composite `at::native::split_with_sizes_copy_out`.
* The outer gate no longer requires the outputs themselves to be
  non-overlapping-and-dense before resizing; the layout check happens after
  resizing, matching CUDA.

## Root cause analysis

`split_with_sizes_copy_out_xpu_kernel` (src/ATen/native/xpu/sycl/TensorShapeKernels.cpp)
entered `split_with_sizes_copy_out_xpu_contiguous_no_cast` whenever
`self.is_non_overlapping_and_dense()` and all outputs were dense with a
matching dtype. That condition does not imply the memory order of the split
dimension matches the trailing dimensions: e.g. `x.t()` for a contiguous `x`
is dense, but `stride(0) == 1` while the trailing numel is `3`. The kernel
then reads `split_size * stride(dim)` elements per chunk - both the wrong
bytes and too many bytes.

Upstream reference: pytorch commit `b0c56ca7`
(aten/src/ATen/native/cuda/TensorShape.cu, test/test_torch.py::test_split_with_sizes_copy_out_non_canonical_layouts,
which currently carries `@skipIfXpu(msg="XPU split_with_sizes_copy.out still
uses stride(dim) as the chunk size")` - this change removes the reason for
that skip).

## Validation (Arc B580, upstream main @ a6aeb5ec + this patch)

Before -> after on the reproducer that mirrors the upstream test:

| case | before | after |
|------|--------|-------|
| dense-but-transposed input, split dim 0 | WRONG values | PASS |
| size-1 split dim, arbitrary stride (guard tail) | OOB write clobbers guard | PASS, guard intact |
| channels-last input -> contiguous outputs | WRONG values | PASS |
| channels-last input -> channels-last outputs | PASS | PASS |
| permuted leading dims, same-permuted outputs | PASS | PASS |
| as_strided (2,10,2)/(1,2,20), split dim 1 | PASS | PASS |
| single non-empty split, transposed layout | PASS | PASS |

* new regression test `test/regressions/test_split_with_sizes_copy_non_canonical.py`: 1/1 PASS
* official `test_torch_xpu.py -k "split"`: test_split_with_sizes_copy_out_xpu,
  test_split_neg_dim PASS
