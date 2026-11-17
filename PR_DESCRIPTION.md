# [pooling] Narrow pooling args via c10::safe_conv like CUDA

Fixes #6031

## Summary

Upstream migrated all CPU/CUDA pooling argument narrowing from `Pool.h`
`safe_downcast` to named `c10::safe_conv` conversions
(pytorch#201921, then pytorch#202291). On overflow CUDA now reports the
destination type, e.g. `value cannot be converted to type int without
overflow`, while the XPU kernels still go through `safe_downcast`, which
reports the opaque placeholder `dest_t`. Pooling tests that match the CUDA
message fail on XPU:

```
FAILED test_MaxPool3d_errors_xpu - expected "value cannot be converted to type int without overflow",
  got "value cannot be converted to type dest_t without overflow"
FAILED test_pooling_backward_overflow_xpu - ... same for dilation
```

## Root cause

The XPU pooling kernels narrow kernel/stride/padding/dilation through
`safe_downcast<int, int64_t>(...)` (`Pool.h`), whose overflow message uses
the default placeholder name (`dest_t`). CUDA narrows the same arguments
through `c10::safe_conv<int>(x, "int")` (`aten/src/ATen/native/cuda/
{DilatedMaxPool2d,DilatedMaxPool3d,AveragePool2d,AveragePool3d}.cu`), whose
message names the destination type.

## Fix

Convert every `safe_downcast<int, int64_t>` argument narrowing in the XPU
pooling kernels to the same `c10::safe_conv<int>(x, "int")` call the CUDA
kernel uses, and the numel narrowing in avg_pool2d to
`c10::safe_conv<int32_t>(x, "int32_t")` (mirroring cuda/AveragePool2d.cu):
one-to-one with the CUDA call sites, no behavior change other than the
error message.

Files: `xpu/DilatedMaxPool2d.cpp`, `xpu/sycl/{DilatedMaxPool2d,
DilatedMaxPool3d,AveragePool2dKernels,AveragePool3dKernels}.cpp`
(69 conversions).

## Verification (Arc B580, incremental build)

Before: `F.max_pool3d(x, kernel_size=(1<<62,3,3))` ->
`value cannot be converted to type dest_t without overflow`
After:  -> `value cannot be converted to type int without overflow`
(same for the dilation path; max_pool2d / avg_pool2d / avg_pool3d unchanged,
they already reported `int`.)

* `test_pooling.py::TestPoolingNNDeviceXPU::test_MaxPool3d_errors_xpu`:
  FAIL -> PASS.
* Pooling subset (`max_pool2d|avg_pool2d|max_pool3d|avg_pool3d ... and xpu`):
  52 passed / 96 pre-existing skips / 0 failures.

Note: the local pinned nightly still carries the pre-pytorch#202291
`c10::safe_conv` message wording ("safely converted ... : int"); against
current pytorch main the same call sites produce "value cannot be converted
to type int without overflow", exactly matching CUDA and the current tests.
