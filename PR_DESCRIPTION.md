# [XPU] Make ctc_loss_backward deterministic-capable (port of pytorch@56ac168)

Fixes #6259

## Summary

Upstream pytorch/pytorch@56ac168 (#201617, fixing pytorch/pytorch#178052) made the
CUDA native CTC loss backward deterministic: under `torch.use_deterministic_algorithms(True)`
the kernel skips the atomicAdd-based large-alphabet path, uses the naive per-element
gradient algorithm, and no longer calls `alertNotDeterministic("ctc_loss_backward_gpu")`.

The XPU kernel (`src/ATen/native/xpu/sycl/LossCTCKernels.cpp`) still mirrored the old
CUDA structure, so `F.ctc_loss(...).backward()` on XPU always raised:

```
RuntimeError: ctc_loss_backward_xpu does not have a deterministic implementation, but you set
'torch.use_deterministic_algorithms(True)'.
```

This PR ports the same three changes to the XPU kernel (the naive `ctc_loss_backward_collect_kernel`
path is already atomic-free, so no new kernel is needed):

- `if (is_large)` → `if (is_large && !globalContext().deterministicAlgorithms())`
- comment on the else branch updated to match CUDA
- unconditional `alertNotDeterministic("ctc_loss_backward_xpu")` removed

It also mirrors the test-side changes of the upstream commit:

- `test/xpu/test_torch_xpu.py`: the inherited `test_nondeterministic_alert_CTCLoss`
  (deleted upstream) is overridden to a skip, guarded by `hasattr` so the override
  disappears once the pinned pytorch moves past 56ac168.
- `test/regressions/test_ctc_loss_deterministic.py`: new regression test mirroring the
  upstream oracle `test_CTCLoss_deterministic` (`test/test_nn.py`): loss bitwise equal
  to the nondeterministic reference, gradient within atol 1e-3 (float) / 1e-11 (double),
  three repeats bitwise identical, and no warning under `warn_only=True`.

## Root cause analysis

`ctc_loss_backward_kernel_template` in the XPU kernel is a line-by-line port of the
pre-56ac168 CUDA implementation. Its `is_large` branch (large alphabet / large batch)
accumulates gradients with `atomicAdd` in `ctc_loss_backward_collect_nonblank_kernel`
(see the `atomicAdd(sycl_global_ptr<scalar_t>(...), -exp_val * gr)` site), hence the
unconditional `alertNotDeterministic`. The small-problem branch
(`ctc_loss_backward_collect_kernel`) computes each output element with no cross-thread
accumulation and is fully deterministic. Upstream's insight applies verbatim: when
deterministic algorithms are requested, take the naive path — it is correct for all
problem sizes, merely slower for large ones.

## Test results (Arc B580, `TORCH_XPU_ARCH_LIST=bmg`, upstream main @ 5dab70aa + this patch)

Before (nightly-equivalent kernel): the issue's oracle raises for all
4 cases × {float32, float64}:

```
small_path torch.float32: raise ctc_loss_backward_xpu does not have a deterministic implementation ...
RESULT: REPRODUCED (8/8)
```

After:

```
small_path torch.float32:            loss_equal=True grad_close=True repeat_bitwise=True max_grad_diff=0.00e+00
long_targets_large_alphabet f32:     loss_equal=True grad_close=True repeat_bitwise=True max_grad_diff=1.50e-04
large_alphabet f32:                  loss_equal=True grad_close=True repeat_bitwise=True max_grad_diff=1.81e-04
boundaries_zero_infinity f32:        loss_equal=True grad_close=True repeat_bitwise=True max_grad_diff=8.42e-06
small_path f64:                      loss_equal=True grad_close=True repeat_bitwise=True max_grad_diff=0.00e+00
long_targets_large_alphabet f64:     loss_equal=True grad_close=True repeat_bitwise=True max_grad_diff=5.76e-13
large_alphabet f64:                  loss_equal=True grad_close=True repeat_bitwise=True max_grad_diff=3.36e-13
boundaries_zero_infinity f64:        loss_equal=True grad_close=True repeat_bitwise=True max_grad_diff=1.47e-13
RESULT: PASS (8/8, warn_only emits no nondeterministic warning)
```

- `pytest test/test_nn.py -k CTCLoss` (upstream suite): 10 passed, 4 skipped, 0 failed.
- The nondeterministic (default) path is untouched: identical results with
  `use_deterministic_algorithms(False)` before/after the patch.

## Notes for reviewers

- The 3-line kernel change is intentionally byte-identical in spirit to the CUDA diff of
  pytorch@56ac168 (`aten/src/ATen/native/cuda/LossCTC.cu`).
- Umbrella intel/torch-xpu-ops#5304 tracks `test_nondeterministic_alert_*` mismatches;
  its CTCLoss entry is resolved by this PR (as the upstream commit resolved the CUDA one).
- Deterministic mode trades speed for reproducibility on large problems (naive path);
  this matches the CUDA behavior decision made upstream.
