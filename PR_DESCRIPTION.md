# Fix fp16 overflow in layer_norm backward grad_input for large normalized dim

Fixes #6310

## Summary

`F.layer_norm` backward on XPU returned `inf` in `grad_input` for `float16`
inputs once the normalized dim N >= 8192 (602 inf elements at (2, 20001), all
131072 at (2, 65536)), while CPU fp16 / CUDA are clean. `grad_weight` and
`grad_bias` were unaffected.

Root cause: in `LayerNormBackward::update()`
(`src/ATen/native/xpu/sycl/LayerNormKernels.cpp`) the leading term was
evaluated as

```cpp
accscalar_t f_grad_input = ... static_cast<accscalar_t>(fH * gamma_val[v] * dY_val[v]);
```

`fH` is an `int` and `at::Half` mixed arithmetic converts the int to `Half`
first, so the product is computed **in float16** and only widened afterwards:

- N = 20001: `Half(fH) * gamma * dY` overflows fp16 range (65504) whenever
  `|gamma * dY| > 3.27` — for the repro's randn inputs that is ~1.5% of
  elements ≈ the observed 602/40002.
- N = 65536: `Half(65536)` is already `inf`, so every element overflows —
  matching the issue's table exactly.
- bfloat16 (wider exponent) and float32 are unaffected, consistent with the
  reported dtype dependence.

The fix casts `gamma_val[v]` / `dY_val[v]` to `accscalar_t` before
multiplying, so the whole expression accumulates in float — matching the CUDA
reference kernel (`aten/src/ATen/native/cuda/layer_norm_kernel.cu`), where
`gamma_val` is loaded via `static_cast<T_ACC>` and `fH` is a `T_ACC`.

Since `update()` is shared with the `rms_norm` backward functor, the same
overflow existed there and is fixed by this change as well (verified: fp16
`rms_norm` backward at N = 20001 produced inf before, none after).

## Testing

- New regression test `test/regressions/test_layer_norm_backward_fp16_overflow.py`
  (layer_norm + rms_norm fp16 backward vs float32 CPU reference, N = 8192 /
  20001 / 65536, same tolerance as upstream `test_layer_norm_backward_shapes`):
  - Before (A770, torch 2.16.0.dev20261116+xpu): 2/2 FAIL (`grad_input
    contains inf`), standalone repro shows inf=602 @ N=20001, 131072 @
    N=65536.
  - After (BMG, torch-xpu-ops main a6aeb5ec + this fix): 2/2 PASS, inf=0 for
    all N, fp32/bf16 unchanged.
- `pytest test/test_nn.py -k "(layer_norm or layernorm) and xpu"`: 2 passed.
- grad_weight / grad_bias still match CPU (unchanged code path).
