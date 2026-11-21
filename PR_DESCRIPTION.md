# [XPU] Add deterministic implementation for upsample_bilinear2d_backward

Fixes #6318

## Summary

Port of the CUDA deterministic upsample bilinear backward (the
`upsample_bilinear2d_backward_gather` family in
aten/src/ATen/native/cuda/UpSampleBilinear2d.cu) to XPU.

`upsample_bilinear2d_backward_out_xpu` unconditionally called
`globalContext().alertNotDeterministic(...)`, so running the backward under
`torch.use_deterministic_algorithms(True)` raised

```
RuntimeError: upsample_bilinear2d_backward_out_xpu does not have a
deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'
```

CUDA (non-HIP) solves this with an alternate, atomics-free implementation
that is selected when deterministic algorithms are enabled. This PR ports
that behavior:

* `TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_xpu)` dispatches to the
  new `xpu::upsample_bilinear2d_backward_out_kernel_deterministic` when
  `globalContext().deterministicAlgorithms()`; the unconditional alert is
  removed (same as CUDA non-HIP).
* New `upsample_bilinear2d_backward_gather_kernel<scalar_t, accscalar_t,
  is_channel_last>`: one work item per *input* element; each work item sums
  the bounded window of grad_output pixels that map to it
  (`compute_output_range`), so no atomics are needed and runs are
  bit-reproducible. Handles NCHW and channels-last layouts.
* Host entry mirrors the CUDA template: same-size shortcut via `copy_`,
  channels-last path when `grad_input` is channels-last contiguous,
  contiguous buffer + copy-back otherwise.

## Root cause analysis

The existing XPU backward has three kernels: an input-centric fast path that
only applies for exact integer upscaling (align / not_align kernels) and an
`atomicAdd`-based output-centric fallback for everything else. Because atomics
make the result order-dependent, the op was registered as nondeterministic via
`alertNotDeterministic`, which throws in deterministic mode. CUDA switched the
generic path to a gather formulation (iterate over input pixels, each
collecting its bounded set of contributing output pixels) in
pytorch#203999-era work; the XPU port never had that variant.

Note: `torch.nn.functional.interpolate` (pytorch core,
torch/nn/functional.py) routes non-CUDA devices to the
`_upsample_linear_vec` decomposition while deterministic algorithms are
enabled, so the alert is only observable when the forward graph is built
outside the deterministic guard (exactly what
`TestNNDeviceTypeCUDA::test_upsamplingBiMode2d_deterministic` does, the test
tracked by this issue).

## Validation (Arc B580, upstream main @ a6aeb5ec + this patch)

* direct op call under `use_deterministic_algorithms(True)`: RAISE -> PASS
  (no alert, correct values)
* `test_upsamplingBiMode2d_deterministic` matrix (scale 0.5/1.5/2 x
  align_corners T/F x contiguous/channels_last, grad vs CPU reference):
  12/12 PASS, max abs diff 0 for downsample, <= 2.4e-07 (fp32 accumulation
  order) for upsample; fp64 within 1e-10
* new regression test
  `test/regressions/test_upsample_bilinear2d_backward_deterministic.py`:
  7/7 PASS (bit-exact reruns, channels-last, fp16/bf16/fp32/fp64,
  scale_factor+recompute case)
* official `test_nn_xpu.py -k "upsampling and not antialias"`: 100 passed
* official `test_torch_xpu.py -k deterministic`: 81 passed
  (`test_nondeterministic_alert_interpolate_bilinear` now passes once its
  stale `expectedFailureXPU` marker from #5304 is dropped - upstream main has
  already deleted that test; the marker lives in the pytorch tree, not this
  repo)

Also fixes #6319 (same kernel; see the native-entry/bit-reproducibility cases
in the regression test). The remaining half of #6319 - letting XPU take the
native path from `F.interpolate` instead of the python-level decomposition -
is a one-line gate change in pytorch core (torch/nn/functional.py currently
exempts only non-HIP CUDA) and can land once this kernel is in.
