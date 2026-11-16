# [embedding_bag] Keep 64-bit index values in the backward kernels

**Fixes #6075**

## Summary

The XPU `embedding_bag` backward truncated index **values** to `int` in two
kernels, so for embedding rows `>= 2**31` (the 64-bit indexing path exercised
by `test_embedding_bag_per_sample_weights_large_index`) the kernels addressed
`weight`, `grad_output`, `offset2bag` and `per_sample_weights` through a
wrapped-negative offset:

- `embedding_bag_per_sample_weights_backward_kernel`
  (`src/ATen/native/xpu/sycl/EmbeddingBag.h`):
  `const int embedding_idx = (int)indices[sample_idx];` — `2**31` becomes
  `INT_MIN`, and `weight[weight_stride0 * embedding_idx + ...]` reads far out
  of bounds. The gradient comes back as `0` instead of `3.0`.
- `compute_grad_weight_bags_kernel`
  (`src/ATen/native/xpu/sycl/EmbeddingBackwardKernel.h`): the same truncation
  for `orig_row`, poisoning `offset2bag_ptr[orig_row]`,
  `per_sample_weights_ptr[orig_row * ...]` and `grad_output_row`.

The CUDA kernels these were ported from use `index_t` (= `int64_t` for
int64 indices) throughout (`EmbeddingBag.cu:502-503`), so XPU diverges from
CUDA for every index `>= 2**31`. This PR restores `index_t` in both places.

## Repro (before → after, Arc B580)

The official test scenario — `weight` `(2**31+1, 1)` fp16, `indices=[2**31]`,
`mode="sum"`, `per_sample_weights=ones`:

```
forward out                [[3.0]]       [[3.0]]        (unchanged)
per_sample_weights.grad    0.0           3.0            (fixed)
weight.grad[2**31] *       garbage/OOB   1.0            (fixed)
* with weight.requires_grad_(True)
```

## Testing

- New regression test `test/regressions/test_embedding_bag_large_index.py`
  (XPU port of the upstream test + a `grad_weight` variant); passes after the
  fix, `grad_psw` case fails before.
- `test/regressions/test_embedding_backward_large_segments.py` still passes.
- CPU-vs-XPU sweep over `embedding_bag` `sum`/`mean` × `scale_grad_by_freq`
  × `per_sample_weights` all match.
