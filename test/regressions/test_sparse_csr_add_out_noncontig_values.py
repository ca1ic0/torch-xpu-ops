# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression tests for ``torch.add`` with a sparse CSR ``out=`` whose
``values()`` is a non-contiguous view, and for in-place ``add_`` on such a
tensor, on XPU.

Upstream pytorch/pytorch@125d69e0 (#203228, fixing pytorch/pytorch#201268)
made the CPU and CUDA ``add`` kernels run on a clone with contiguous values
and copy the result back through the view.  The XPU kernel
``add_out_sparse_compressed_xpu`` instead resized ``out`` and replaced its
values, so the buffer behind a strided values view was never written
(intel/torch-xpu-ops#6201).
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


class TestSparseAddOutNoncontiguousValues(TestCase):
    def _full_csr(self, dtype, device):
        return (
            torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=dtype, device=device
            )
            .to_sparse_csr()
            .clone()
        )

    def test_add_out_writes_through_strided_values(self, device):
        for dtype in (torch.float64, torch.float32):
            a = self._full_csr(dtype, device)
            b = a.clone()
            expected = a.to_dense() + 0.5 * b.to_dense()
            out, buf = _with_strided_values(a)
            values_before = out.values()
            torch.add(a, b, alpha=0.5, out=out)
            _check_written_through_view(self, out, buf, expected)
            self.assertEqual(out.values().data_ptr(), values_before.data_ptr())

    def test_add_inplace_writes_through_strided_values(self, device):
        for dtype in (torch.float64, torch.float32):
            a = self._full_csr(dtype, device)
            b = a.clone()
            expected = a.to_dense() + 0.5 * b.to_dense()
            out, buf = _with_strided_values(a)
            out.add_(b, alpha=0.5)
            _check_written_through_view(self, out, buf, expected)

    def test_sampled_addmm_out_writes_through_strided_values(self, device):
        # Same guard as pytorch/pytorch#203228 adds to the CPU and CUDA
        # sampled_addmm out= kernels.
        for dtype in (torch.float64, torch.float32):
            mat1 = torch.randn(4, 6, dtype=dtype, device=device)
            mat2 = torch.randn(6, 5, dtype=dtype, device=device)
            crow = torch.tensor([0, 2, 3, 5, 7], dtype=torch.int64, device=device)
            col = torch.tensor([0, 2, 1, 0, 3, 2, 4], dtype=torch.int64, device=device)
            vals = torch.randn(7, dtype=dtype, device=device)
            c = torch.sparse_csr_tensor(crow, col, vals, size=(4, 5))
            expected = torch.sparse.sampled_addmm(c, mat1, mat2).to_dense()
            out, buf = _with_strided_values(c)
            values_before = out.values()
            torch.sparse.sampled_addmm(c, mat1, mat2, out=out)
            _check_written_through_view(self, out, buf, expected)
            self.assertEqual(out.values().data_ptr(), values_before.data_ptr())


instantiate_device_type_tests(
    TestSparseAddOutNoncontiguousValues,
    globals(),
    only_for="xpu",
    allow_xpu=True,
)

if __name__ == "__main__":
    run_tests()
