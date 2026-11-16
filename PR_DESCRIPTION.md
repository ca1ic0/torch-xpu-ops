# [batch_norm] Validate batch_norm_gather_stats shapes before launching the XPU kernel

**Fixes #6108**

## Summary

`batch_norm_gather_stats_xpu` and `batch_norm_gather_stats_with_counts_xpu`
forwarded straight to the kernel without any shape validation. The kernel
(`batch_norm_reduce_statistics_kernel`) derives both of its loop bounds from
`mean` alone and then indexes `invstd` and `counts` with them, so any
disagreement is an out-of-bounds read — on the reporting machine this
core-dumps the process (#6108); on Arc B580 the same repro silently returns
garbage instead of raising. Either way the expected `RuntimeError` never
reaches the user.

The CUDA wrappers in
[`Normalization.cu`](https://github.com/pytorch/pytorch/blob/2a2d844941bf776d1a49a8ff8730e90f5242f836/aten/src/ATen/native/cuda/Normalization.cu#L700-L735)
guard this exact hazard (added with the regression test from
pytorch/pytorch#189828), with a comment spelling out the OOB risk:

> "batch_norm_reduce_statistics_kernel derives both of its loop bounds from
> mean alone, then indexes invstd and counts with them, so any disagreement
> is an out-of-bounds read."

## Changes

`src/ATen/native/xpu/BatchNorm.cpp`, both entry points, same `TORCH_CHECK`s
and same messages as CUDA:

- `batch_norm_gather_stats_xpu`: `mean.dim() == 2` and
  `invstd.sizes() == mean.sizes()`.
  (CUDA's `batch_norm_gather_stats_cuda` builds a counts tensor and forwards
  to the `_with_counts` path, which is where its invstd check lives; the two
  XPU entry points each launch the kernel directly, so the invstd check is
  added here as well.)
- `batch_norm_gather_stats_with_counts_xpu`: the above plus
  `counts.numel() >= mean.size(0)`.

Host-side pre-checks only; the valid-input path is untouched.

## Testing

Intel Arc B580, pytorch `bb84fa86` (2026-10-26 nightly tree) + torch-xpu-ops
rebuilt with this patch:

```
invstd mismatch: RuntimeError -> batch_norm_gather_stats_with_counts: expected invstd to
    have the same shape as mean ([2, 3]), but got invstd of sizes [2, 0]
1-D mean:        RuntimeError -> batch_norm_gather_stats: expected mean to be 2-dimensional
    (world_size, num_features), but got mean of sizes [3]
short counts:    RuntimeError -> batch_norm_gather_stats_with_counts: expected counts to have
    at least one element per entry in mean first dimension (2), but got 1 elements
valid call OK: (3,) (3,)
```

- The three messages match the three `assertRaisesRegex` patterns in
  `test_nn.py::TestNNDeviceType::test_batch_norm_gather_stats_invalid_shapes`
  (the pytorch#189828 regression test; its XPU variant is skipped pending
  this issue and can be un-skipped alongside).
- `test/xpu/test_nn_xpu.py -k batch_norm`: passes, valid paths unaffected.

- [x] all three invalid shapes raise the same RuntimeError as CUDA
- [x] valid calls unchanged
- [ ] CI: un-skip `test_batch_norm_gather_stats_invalid_shapes_xpu` once available
