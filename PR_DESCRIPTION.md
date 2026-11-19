# [XPU][Sparse] Write mm/addmm out= results through non-contiguous values views

Fixes #6173

## Summary

`torch.mm(a, b, out=out)` and `torch.addmm(c, a, b, out=out)` with a sparse
CSR `out` whose `values()` is a non-contiguous view did not write the result
through that view on XPU (`mm.out` reaches
`addmm_out_sparse_compressed_xpu` via `_sparse_csr_mm_out`). This adds the
guard pytorch/pytorch#203228 (pytorch/pytorch@125d69e0) added to the CPU and
CUDA `addmm` kernels, using the helpers from that PR:

```cpp
if (at::sparse_csr::has_noncontiguous_values(result)) {
  return at::sparse_csr::run_on_contiguous_clone(result, [&](Tensor& r) {
    addmm_out_sparse_compressed_xpu(self, mat1, mat2, beta, alpha, r);
  });
}
```

Then the `skipXPUIf` on `test_mm_addmm_out_noncontiguous_values` in
pytorch's `test/test_sparse_csr.py` can be removed.

## Root cause

`addmm_out_sparse_compressed_xpu` resizes `result` and passes it to the Sycl
kernels, which write `result.values()` through `data_ptr()` assuming
contiguous values. With a strided values view, the kernel wrote a fresh
contiguous tensor instead, leaving the buffer behind the view untouched.

## Reproducer (before → after)

```python
import torch
device = "xpu"
a = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64, device=device).to_sparse_csr()
expected = a.to_dense() @ a.to_dense()
buf = torch.full((8,), -99., dtype=torch.float64, device=device)
out = torch.sparse_csr_tensor(a.crow_indices(), a.col_indices(), buf[::2], size=(2, 2))
torch.mm(a, a, out=out)
print("ok" if torch.equal(out.to_dense(), expected) and (buf[1::2] == -99).all() else "WRONG")
```

Before: `WRONG` (buf untouched). After: `ok`, matching CPU/CUDA with
pytorch#203228. Verified on BMG (B580, torch-xpu-ops main 427e033a,
incremental rebuild): `mm out=` and `addmm out=` write through the view for
float64/float32 (2 new regression tests pass); the existing
`test/test_sparse_csr.py -k "mm or matmul or sample"` subset still passes
(154 passed, 0 failed).

## Testing

- `pytest test/regressions/test_sparse_csr_mm_addmm_out_noncontig_values.py` (new)
- `test/xpu/test_sparse_csr_xpu.py -k "mm or addmm"` subset
