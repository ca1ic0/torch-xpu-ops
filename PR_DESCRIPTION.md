# [XPU][Sparse] Write add out= results through non-contiguous values views

Fixes #6201

## Summary

`torch.add(a, b, out=out)` / `out.add_(b)` with a sparse CSR `out` whose
`values()` is a non-contiguous view replaced `out.values()` on XPU instead of
writing the result through that view, diverging from CPU and CUDA since
pytorch/pytorch#203228 (pytorch/pytorch@125d69e0).

This adds the same guard the upstream PR added to the CPU and CUDA `add` and
`sparse_sampled_addmm` `out=` kernels: when `out` has non-contiguous values,
run on a clone with contiguous values and copy the result back through the
view, via the `at::sparse_csr::has_noncontiguous_values` /
`run_on_contiguous_clone` helpers that PR added to
`ATen/SparseCsrTensorUtils.h`:

```cpp
if (at::sparse_csr::has_noncontiguous_values(out)) {
  return at::sparse_csr::run_on_contiguous_clone(out, [&](Tensor& r) {
    add_out_sparse_compressed_xpu(self, other, alpha, r);
  });
}
```

The same guard is added to `sparse_sampled_addmm_out_sparse_csr_xpu` for
parity with the CPU/CUDA kernels (its XPU tests pass today, but the kernel
has the same latent issue).

The guard makes the in-place `out.add_(b)` path feed the strided `self`
values to `add_out_sparse_csr_kernel` as an *input*, and the kernel read
`A.values().to(dtype)` raw pointers — a no-op `.to()` keeps the strided
view, so the reads walked the wrong storage. The kernel now materializes
non-contiguous input values with `.contiguous()` before taking pointers,
matching upstream's accessor-based CPU reads.

After this, the XPU-side skip of `test_sparse_add_aliased_out`'s
non-contiguous-values part (pytorch/pytorch#204912) can be lifted.

## Root cause

`add_out_sparse_compressed_xpu` (and `sparse_sampled_addmm_out_sparse_csr_xpu`)
call `resize_as_sparse_compressed_` and then hand `out` to the Sycl kernel,
which writes `out.values()` through `data_ptr()` assuming a contiguous
values tensor. When `out.values()` is a strided view, `resize_`/kernel writes
replaced the view's storage, so the buffer behind the view was never updated.
Secondarily, `add_out_sparse_csr_kernel` read input values through raw
pointers as well, so strided input values (the in-place path under the new
guard) produced wrong sums until materialized contiguously.

## Reproducer (before → after)

```python
import torch
device = "xpu"
a = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64, device=device).to_sparse_csr()
b = a.clone()
buf = torch.full((8,), -99., dtype=torch.float64, device=device)
out = torch.sparse_csr_tensor(a.crow_indices(), a.col_indices(), buf[::2], size=(2, 2))
torch.add(a, b, out=out)
print("ok" if torch.equal(buf[::2], 2 * a.values()) and (buf[1::2] == -99).all() else "WRONG")
```

Before: `WRONG` (buf untouched; `out.values()` was replaced by a fresh
contiguous tensor). After: `ok`, matching CPU/CUDA with pytorch#203228.
Verified on BMG (B580, torch-xpu-ops main 427e033a, incremental rebuild):
`add out=`, in-place `add_`, and `sparse_sampled_addmm out=` write through
the view for float64/float32 (3 new regression tests pass); the existing
`test/test_sparse_csr.py -k add` subset still passes (167 passed, 0 failed).

## Testing

- `pytest test/regressions/test_sparse_csr_add_out_noncontig_values.py` (new)
- `test/xpu/test_sparse_csr_xpu.py -k "add or sampled_addmm"` subset
