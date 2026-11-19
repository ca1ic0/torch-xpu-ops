# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression tests for ``torch.mm`` / ``torch.addmm`` with a sparse CSR
``out=`` whose ``values()`` is a non-contiguous view, on XPU.

Upstream pytorch/pytorch@125d69e0 (#203228, fixing pytorch/pytorch#201268)
made the CPU and CUDA ``addmm`` kernels run on a clone with contiguous
values and copy the result back through the view.  The XPU kernel
``addmm_out_sparse_compressed_xpu`` (which also serves ``mm.out`` via
``_sparse_csr_mm_out``) instead ignored the view, so the buffer behind a
strided values view was never written (intel/torch-xpu-ops#6173).
"""

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase

FILL = -99.0


def _with_strided_values(t, fill=FILL):
    """Return a CSR tensor with t's sparsity pattern whose values are a
    stride-2 view into a buffer twice as long, plus the buffer."""
    values = t.values()
    buf = torch.full(
        (2 * values.numel(),), fill, dtype=values.dtype, device=values.device
    )
    strided_values = buf[::2].view(values.shape)
    strided_values.copy_(values)
    out = torch.sparse_csr_tensor(
        t.crow_indices().clone(), t.col_indices().clone(), strided_values, size=t.shape
    )
    return out, buf


def _check_written_through_view(test, out, buf, expected, fill=FILL):
    test.assertEqual(out.to_dense(), expected)
    test.assertEqual(buf[::2].flatten(), out.values().flatten())
    test.assertEqual(buf[1::2], torch.full_like(buf[1::2], fill))


class TestSparseMmAddmmOutNoncontiguousValues(TestCase):
    def _full_csr_2x2(self, dtype, device):
        return (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, device=device)
            .to_sparse_csr()
            .clone()
        )

    def test_mm_out_writes_through_strided_values(self, device):
        # Issue reproducer: full 2x2 CSR, so mm(a, a) keeps nnz 4 and the
        # result must go through the stride-2 values view.
        for dtype in (torch.float64, torch.float32):
            a = self._full_csr_2x2(dtype, device)
            expected = a.to_dense() @ a.to_dense()
            out, buf = _with_strided_values(a)
            values_before = out.values()
            torch.mm(a, a, out=out)
            _check_written_through_view(self, out, buf, expected)
            self.assertEqual(out.values().data_ptr(), values_before.data_ptr())

    def test_addmm_out_writes_through_strided_values(self, device):
        # As in upstream test_mm_addmm_out_noncontiguous_values: out keeps the
        # result's nnz, so the result must be written through the values view.
        # Full (no structural zeros) matrices make every nnz equal.
        for dtype in (torch.float64, torch.float32):
            torch.manual_seed(0)
            a = torch.randn(4, 6, dtype=dtype, device=device).to_sparse_csr()
            b = torch.randn(6, 5, dtype=dtype, device=device).to_sparse_csr()
            c = torch.randn(4, 5, dtype=dtype, device=device).to_sparse_csr()
            expected = 0.5 * c.to_dense() + 2 * (a.to_dense() @ b.to_dense())
            out, buf = _with_strided_values(c)
            values_before = out.values()
            torch.addmm(c, a, b, beta=0.5, alpha=2, out=out)
            _check_written_through_view(self, out, buf, expected)
            self.assertEqual(out.values().data_ptr(), values_before.data_ptr())


instantiate_device_type_tests(
    TestSparseMmAddmmOutNoncontiguousValues,
    globals(),
    only_for="xpu",
    allow_xpu=True,
)

if __name__ == "__main__":
    run_tests()
