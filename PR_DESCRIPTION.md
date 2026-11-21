# [XPU] Add deterministic implementation for upsample_bilinear2d_backward (interpolate native path)

Fixes #6319

## Summary

Port of the CUDA deterministic upsample bilinear backward (the
`upsample_bilinear2d_backward_gather` family in
aten/src/ATen/native/cuda/UpSampleBilinear2d.cu) to XPU, so that
`F.interpolate(..., mode="bilinear")` backward can run under
`torch.use_deterministic_algorithms(True)` natively.

`test_deterministic_interpolate_bilinear` (test_torch.py) requires that under
deterministic mode the backward

1. does not raise,
2. is bit-reproducible across reruns,
3. matches the CPU reference within dtype tolerance,
4. supports channels-last (the nhwc_vec cases) and the
   scale_factor+recompute case,
5. does **not** fall back to the `torch._decomp.decompositions._upsample_linear_vec`
   decomposition (the test patches it to raise).

XPU had no native deterministic backward: `upsample_bilinear2d_backward_out_xpu`
unconditionally alerted nondeterminism and its generic kernel is
`atomicAdd`-based. This PR ports the CUDA gather formulation:

* dispatch to the new deterministic kernel when
  `globalContext().deterministicAlgorithms()` (alert removed, matching
  CUDA non-HIP);
* `upsample_bilinear2d_backward_gather_kernel<scalar_t, accscalar_t,
  is_channel_last>`: one work item per input element, bounded output window
  via `compute_output_range`, no atomics - deterministic by construction,
  NCHW and channels-last layouts, fp16/bf16/fp32/fp64.

## Root cause analysis

See the twin branch `fix/upsample-bilinear2d-backward-deterministic` (issue
#6318) for the full analysis; both issues are fixed by the same kernel. What
#6319 additionally exposes:

* the nhwc_vec cases exercise channels-last layouts with several dtypes -
  covered by the channels-last gather kernel variant;
* the scale_factor_recompute case exercises recomputed output sizes -
  covered by the size-based host path;
* criterion 5 (no decomposition) additionally needs a pytorch-core gate
  change: `torch/nn/functional.py` currently exempts only non-HIP CUDA from
  the `_upsample_linear_vec` deterministic fallback
  (`not input.is_cuda and torch.version.hip is None`). XPU should be added
  there once this kernel has landed; with the alert removed, routing XPU to
  the native op would otherwise be the only missing piece.

## Validation (Arc B580, upstream main @ a6aeb5ec + this patch)

* native entry (`torch._C._nn.upsample_bilinear2d`) forward + backward under
  `DeterministicGuard(True)`:
  * (1,8,4,5)->(8,10) channels-last, fp16/bf16/fp32/fp64: bit-exact reruns,
    matches CPU reference within dtype tolerances
  * (2,3,4,5)->(8,10) contiguous: same
  * scale_factor=(1.7,2.0) recompute case: bit-exact reruns, within 1e-5 of
    the CPU reference
* new regression test
  `test/regressions/test_upsample_bilinear2d_backward_deterministic.py`:
  7/7 PASS
* official `test_nn_xpu.py -k "upsampling and not antialias"`: 100 passed;
  `test_torch_xpu.py -k deterministic`: 81 passed (after dropping the stale
  `expectedFailureXPU` marker for test_nondeterministic_alert_interpolate_bilinear
  in the pytorch tree; upstream main already deleted that test)
