# [XPU][Sparse] Handle strided COO indices in _convert_indices_from_coo_to_csr

Fixes #6277

## Summary

`torch._convert_indices_from_coo_to_csr` with a **non-contiguous (strided)** input
returns a wrong `crow` on XPU (first reported on Windows/LNL; reproduced and fixed on
Linux/BMG — the defect is platform-independent):

```
row = torch.tensor([[0, 99], [1, 88], [2, 77]], dtype=torch.int64)[:, 0]   # [0, 1, 2], stride 2
crow = torch._convert_indices_from_coo_to_csr(row, size=4)
# expected [0, 1, 2, 3, 3], got [0, 1, 1, 1, 1]
```

This ports upstream pytorch/pytorch@9355db71 (#202883, fixing pytorch/pytorch#202122),
which fixed the identical defect on CPU and CUDA: take
`input.expect_contiguous()` before reading the raw data pointer. The same test that
upstream added (`test_coo_to_csr_convert_noncontiguous_indices`) is mirrored as a
regression test here.

## Root cause analysis

`launch_convert_indices_from_coo_to_csr_xpu_kernel`
(`src/ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.cpp`) reads the input
through a raw pointer:

```cpp
const input_t* data_in = input.const_data_ptr<input_t>();
```

and `convert_indices_from_coo_to_csr_xpu_kernel` then walks `data_in[linear_id]`
linearly, assuming a stride-1 layout. For `t[:, 0]` of a contiguous `(3, 2)` tensor
(elements `[0, 1, 2]`, stride 2) the kernel instead reads the flat storage
`[0, 99, 1]`. Treating `99` as a row index makes the `linear_id == 1` work item run
`for (i = 0; i < 99; i++) data_out[i + 1] = 1;` — writing ~99 elements into a
5-element output (out-of-bounds GPU writes; the visible symptom is the corrupted
`crow`). Upstream's `expect_contiguous()` fix applies verbatim: contiguous inputs
borrow the tensor; strided ones get materialized first.

## Test results (Arc B580, `TORCH_XPU_ARCH_LIST=bmg`, upstream main @ 5dab70aa + this patch)

Before:

```
case1 out_int32=False: crow=[0, 1, 1, 1, 1] expect=[0, 1, 2, 3, 3] -> FAIL
case1 out_int32=True:  crow=[0, 1, 1, 1, 1] expect=[0, 1, 2, 3, 3] -> FAIL
case2: to_sparse_csr=False to_sparse_bsr=False mm=True
```

After:

```
case1 out_int32=False: crow=[0, 1, 2, 3, 3] expect=[0, 1, 2, 3, 3] -> OK
case1 out_int32=True:  crow=[0, 1, 2, 3, 3] expect=[0, 1, 2, 3, 3] -> OK
case2: to_sparse_csr=True to_sparse_bsr=True mm=True
```

- `pytest test/xpu/test_sparse_csr_xpu.py` (full official suite): **8372 passed,
  1210 skipped, 4 xfailed, 0 failed**.
- `test/regressions/test_coo_to_csr_strided_indices.py`: all 3 tests pass.

## Notes for reviewers

- The two-line kernel change is byte-equivalent to the CPU/CUDA change in
  pytorch@9355db71 (`expect_contiguous()`).
- The sibling `convert_indices_from_csr_to_coo` kernel also uses raw pointers, but
  upstream did not change it in #202883 (crow/col indices of CSR tensors are
  internally contiguous), so this PR stays minimal and mirrors upstream exactly.
