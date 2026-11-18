# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression tests for strided (non-contiguous) COO indices in
torch._convert_indices_from_coo_to_csr on XPU.

Upstream pytorch/pytorch@9355db71 (#202883, fixing pytorch/pytorch#202122)
fixed wrong CSR results for strided COO indices on CPU and CUDA by calling
``input.expect_contiguous()`` before taking the raw data pointer.  The XPU
launcher kept reading ``input.const_data_ptr()`` directly, so a strided
row-index tensor made the kernel walk the underlying flat storage and
produce a wrong ``crow`` (and, for large index values, out-of-bounds
``data_out`` writes).
"""

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCooToCsrStridedIndices(TestCase):
    def test_coo_to_csr_noncontiguous_row_indices(self, device):
        # Sliced column of an (N, 2) matrix: elements [0, 1, 2] with stride 2.
        row_noncontig = torch.tensor(
            [[0, 99], [1, 88], [2, 77]], device=device, dtype=torch.int64
        )[:, 0]
        self.assertFalse(row_noncontig.is_contiguous())
        for out_int32, index_dtype in ((False, torch.int64), (True, torch.int32)):
            crow = torch._convert_indices_from_coo_to_csr(
                row_noncontig, size=4, out_int32=out_int32
            )
            expect = torch.tensor([0, 1, 2, 3, 3], dtype=index_dtype, device=device)
            self.assertEqual(crow, expect)

    def test_coo_to_csr_transposed_coordinate_matrix(self, device):
        # A coalesced COO tensor whose (2, nnz) indices are a transposed view.
        coords_t = torch.tensor(
            [[0, 2], [1, 0], [1, 3], [2, 1]], device=device, dtype=torch.int64
        ).t()
        self.assertFalse(coords_t.is_contiguous())
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device=device)
        coo_noncontig = torch.sparse_coo_tensor(
            coords_t, vals, (3, 4), is_coalesced=True
        )
        self.assertFalse(coo_noncontig.indices().is_contiguous())
        dense = coo_noncontig.to_dense()
        self.assertEqual(coo_noncontig.to_sparse_csr().to_dense(), dense)
        self.assertEqual(coo_noncontig.to_sparse_bsr((1, 1)).to_dense(), dense)
        self.assertEqual(
            torch.mm(coo_noncontig, coo_noncontig.t()).to_dense(), dense @ dense.t()
        )

    def test_coo_to_csr_contiguous_unchanged(self, device):
        # Contiguous inputs must keep working.
        row = torch.tensor([0, 0, 1, 3, 3], device=device, dtype=torch.int64)
        crow = torch._convert_indices_from_coo_to_csr(row, size=5, out_int32=False)
        self.assertEqual(
            crow, torch.tensor([0, 2, 3, 3, 5, 5], dtype=torch.int64, device=device)
        )


instantiate_device_type_tests(
    TestCooToCsrStridedIndices, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
