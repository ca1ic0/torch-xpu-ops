# [segment_reduce] Accumulate in the opmath type to match upstream CPU/CUDA

**Fixes #6181**
**Fixes #6182**

## Summary

The XPU `segment_reduce` forward kernel keeps its running accumulator in
`scalar_t`, so `float16` / `bfloat16` inputs are also *accumulated* at
16-bit precision. Both upstream backends accumulate in
`at::opmath_type<scalar_t>` (float):

- CUDA always has: [`SegmentReduce.cu:102`](https://github.com/pytorch/pytorch/blob/2a2d844941bf776d1a49a8ff8730e90f5242f836/aten/src/ATen/native/cuda/SegmentReduce.cu#L102) — `at::opmath_type<scalar_t> initial_value`
- CPU since [`1c720466`](https://github.com/pytorch/pytorch/commit/1c7204666b9282c1b00e1fabad2b276598a6c5ef) (PR pytorch/pytorch#204079, 2026-11-11)

Once the CPU reference switched to float accumulation, the XPU-vs-CPU
comparison in `test_compare_cpu__segment_reduce_{lengths,offsets}_xpu_{float16,bfloat16}`
started failing (#6181, first bad nightly 2026-11-12, exactly after
`1c720466` landed), and `test_reduced_precision_accumulation` fails for
the same reason (#6182: summing 4096 reduced-precision values is off by
3840.0 absolute vs a float accumulation).

This PR aligns the XPU forward kernel with the CUDA implementation 1:1:
the accumulator lives in `at::opmath_type<scalar_t>` end to end (kernel
parameter, per-element load, host-side initial value) and is narrowed
back to `scalar_t` once on write-back.

## Root cause

`src/ATen/native/xpu/sycl/SegmentReduceKernels.cpp`:

- kernel parameter `scalar_t initial_value_raw` (forward kernel + impl),
- `initial_value = initial_value + data` with both operands in `scalar_t`,
- host side `scalar_t initial_value = 0; initial.value().to<scalar_t>()`.

Each partial sum of a fp16/bfloat16 segment is rounded to 16-bit
mantissa; over thousands of elements the error grows without bound
relative to the float-accumulated reference. The kernel already used
`at::opmath_type<scalar_t>` in the `sycl::isnan(...)` probes, so the
wider type was available — the accumulation main path just never aligned
with CUDA.

## Changes

All in `src/ATen/native/xpu/sycl/SegmentReduceKernels.cpp` (forward path
only, same scope as upstream `1c720466`):

1. `segment_reduce_forward_kernel` gains a dedicated `accscalar_t` template
   parameter (instantiated with `at::opmath_type<scalar_t>` on the host) and
   takes `accscalar_t initial_value_raw`. A dedicated template parameter is
   required because the SYCL free-function kernel extractor cannot resolve
   `at::opmath_type<scalar_t>` spelled out in the kernel signature (it gets
   rewritten to an undefined `at::OpMathType` in the generated device
   header) — this mirrors the existing
   `upsample_bicubic2d_backward_kernel<scalar_t, accscalar_t>` convention.
2. per-element load widened: `static_cast<accscalar_t>(values_data[data_index])`
   (MAX/MIN `isnan` probes and `std::max`/`std::min` no longer need
   explicit casts / forced template argument)
3. host side (`segment_reduce_forward_kernel_impl` + dispatch lambda):
   `using opmath_t = at::opmath_type<scalar_t>;` for the initial value
   (`initial.value().to<opmath_t>()`, `std::numeric_limits<opmath_t>` ±inf,
   `NAN` in opmath), kernel launch templates updated to
   `<scalar_t, opmath_t, index_t>`
4. write-back narrowed once: `static_cast<scalar_t>(initial_value)`

`float32` inputs are unaffected (`opmath_type<float> == float`, bit-identical
behavior). MIN/MAX results pass through float losslessly; PROD now matches
CUDA's behavior for reduced precision dtypes. The backward kernel is
unchanged, mirroring the scope of the upstream CPU fix.

## Testing

Minimal reproducer (sum of 4096 randn values, single segment, XPU result
vs float32 accumulation of the same data), Intel Arc B580, torch
2.16.0.dev nightly + rebuilt torch-xpu-ops (pytorch `bb84fa86`, oneAPI
2026.1):

| dtype | variant | before | after |
|---|---|---|---|
| float16 | lengths | 0.363152 | 0.019402 |
| float16 | offsets | 0.426235 | 0.003452 |
| bfloat16 | lengths | 9.590355 | 0.090355 |
| bfloat16 | offsets | 1.700714 | 0.049286 |

The residual error after the fix is the single fp16/bf16 rounding of the
final result plus parallel-reduction ordering vs the scalar reference —
the same behavior class as the CUDA backend.

```
PYTORCH_TEST_WITH_SLOW=1 python test/xpu/test_segment_reductions_xpu.py   # 74 passed (fp32/16/bf16, sum/mean/min/max/prod, fwd+bwd)
```

The four `test_compare_cpu__segment_reduce_*` cases (#6181) pass once the
CPU side is at pytorch >= `1c720466` (both sides then accumulate in float);
on the older 2026-10-26 tree used for local verification the mismatch
direction simply flips (XPU now closer to the float reference than the old
scalar_t-accumulating CPU: XPU −64.00 vs CPU −67.00 vs float ref −64.19).

- [x] minimal reproducer converges to the float reference on B580
- [x] full `test_segment_reductions_xpu.py` no regression (74 passed)
- [ ] CI: `test_reduced_precision_accumulation_xpu_{float16,bfloat16}` (#6182)
- [ ] CI: `test_compare_cpu__segment_reduce_{lengths,offsets}_xpu_{float16,bfloat16}` (#6181)
