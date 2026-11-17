# [sparse] Fix SIGFPE in the sparse binary op intersection coalesce heuristic

Fixes #6199

## Summary

`sparse * sparse` (COO) kills the process with SIGFPE on XPU when a sparse dim
is zero-sized, or when `prod(shape[:sparse_dim])` overflows int32 (e.g.
65536 x 65536). Port of pytorch/pytorch#204266 (landed on pytorch main as
`b31edb6f558`, in `aten/src/ATen/native/sparse/SparseBinaryOpIntersectionCommon.h`),
which fixed the identical bug in the CPU/CUDA heuristic.

## Root cause

`src/ATen/native/sparse/xpu/sycl/SparseBinaryOpIntersectionKernels.cpp`
computes the coalesce heuristic lower bound on the host:

```cpp
const auto sparse_dim_numel = std::accumulate(
    larger_sizes.begin(), larger_sizes.begin() + larger.sparse_dim(),
    1,                                   // <-- int accumulator
    std::multiplies<int64_t>());
const auto max_count_lower_bound = larger._nnz() / sparse_dim_numel;  // div by 0
```

* The init value `1` makes `std::accumulate` keep the product in `int`, so
  sizes like 65536 x 65536 wrap to 0.
* A zero-size sparse dim yields a 0 product directly.
* The following integer division by 0 raises SIGFPE and takes down the whole
  process (host-side signal, not a catchable RuntimeError).

## Fix (mirrors pytorch@b31edb6f558)

* Accumulate in `int64_t`: pass `static_cast<int64_t>(1)` as the init value.
* Guard the division: a zero-size sparse dim admits no valid indices, so
  there is nothing to coalesce - use `0` as the lower bound.

## Reproducer (both SIGFPE before, pass after)

```python
import torch
a = torch.zeros(0, 3, device="xpu").to_sparse(sparse_dim=1)
b = torch.zeros(0, 1, device="xpu").to_sparse(sparse_dim=1)
a * b  # SIGFPE before; empty (0, 3) result after

i = torch.tensor([[0, 1], [0, 1]], device="xpu")
a = torch.sparse_coo_tensor(i, torch.ones(2, 3, device="xpu"), (2**16, 2**16, 3))
b = torch.sparse_coo_tensor(i, torch.ones(2, 1, device="xpu"), (2**16, 2**16, 1))
a * b  # SIGFPE before; values of ones at indices i after
```

## Verification (Arc B580, torch 2.16.0.dev nightly + this patch, incremental build)

* Both reproducers: rc=-8 (SIGFPE) before, rc=0 with correct results after
  (case 1: shape (0, 3) empty; case 2: coalesced values match CPU).
* `a * b` cross-checked against CPU for float32/float16/bfloat16/float64/
  int32/complex64/complex128: all pass (these include the bf16/complex128
  variants that pytorch#204889 / pytorch#204911 had to disable on XPU).
* No torch-xpu-ops test regressions in the sparse test subset.

Once this lands, the XPU skips added by pytorch#204889 / pytorch#204911 can
be dropped from `third_party/xpu.txt` and those issues closed.
