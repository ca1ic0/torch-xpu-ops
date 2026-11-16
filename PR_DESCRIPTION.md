# [grid_sampler] Accumulate the backward grid gradient in opmath_t

**Fixes #6162**

## Summary

The 2D/3D bilinear and 2D bicubic `grid_sampler` backward kernels kept the
grid-gradient accumulators (`gix`/`giy`/`giz`) in `scalar_t`, so for
float16/bfloat16 inputs the per-channel sum stopped growing once a tap fell
below half a ULP of the running sum. With 2048 channels that each contribute
exactly 1 to the x-gradient (a value exactly representable in both dtypes):

| dtype   | before        | after (this PR) |
|---------|---------------|-----------------|
| float16 | [1024, 0]     | [2048, 0]       |
| bfloat16| [128, 0]      | [2048, 0]       |
| float32 | [2048, 0]     | [2048, 0]       |

The 3D bicubic backward kernel in the same file already accumulates in
`at::opmath_type<scalar_t>` and casts once at the store
(`GridSampler.cpp:1619-1667`); this PR brings the three remaining kernels to
that style: accumulators widened to `opmath_t`, `gOut`/input tap loads
widened, one cast at the `grad_grid` store. The `grad_input` atomic-add path
is unchanged numerically (`safe_add_2d/3d` deltas are computed in fp32 and
rounded back to `scalar_t`, which is bitwise the previous `scalar_t` product).

This matches the direction of the upstream fix for the same defect
(pytorch/pytorch#204495, open at time of writing, scoped to ROCm). CUDA today
also accumulates in `scalar_t`; XPU's own 3D bicubic kernel demonstrates the
intended style in this codebase, and half-precision grid gradients being up
to 16x too small silently stalls training of spatial transformers /
deformable sampling on XPU.

## Testing

- New regression test
  `test/regressions/test_grid_sampler_backward_accumulate.py` (exact-value
  check from the issue; passes after, fails before).
- 40-config sweep (fp16/bf16 × {bilinear, nearest, bicubic} × align_corners ×
  {2D, 3D} × {zeros, border}): max XPU-vs-fp64-CPU error after the fix never
  exceeds 1.5x+1e-3 of the baseline error (0 regressions); mean grid-grad
  error slightly improves (6.577 → 6.539). fp32/fp64 match CPU exactly.
