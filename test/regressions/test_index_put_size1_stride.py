# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestIndexPutSize1Stride(TestCase):
    def _check(self, accumulate):
        # Regression test for torch-xpu-ops issue #6084 (upstream fix on
        # pytorch main for CUDA, pytorch/pytorch#203889): the deterministic
        # index_put_ path stepped between batches with src.stride(i-1), which
        # is arbitrary on a size-1 dim even for a contiguous view, so writes
        # landed outside the indexed tensor.
        for device in ("cpu", "xpu"):
            base = torch.zeros(16, device=device)
            x = base.as_strided((2, 1, 2), (2, 8, 1))
            self.assertTrue(x.is_contiguous())
            idx = torch.tensor([1], device=device)
            val = torch.ones(2, 1, 1, device=device)
            if accumulate:
                x[:, :, idx] += val
                x[:, :, idx] += val
            else:
                x[:, :, idx] = val
            fill = 2.0 if accumulate else 1.0
            self.assertEqual(
                base.cpu(),
                torch.tensor([0, fill, 0, fill, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            )

    def test_index_put_size1_stride_non_accumulate(self):
        torch.use_deterministic_algorithms(True)
        try:
            self._check(accumulate=False)
        finally:
            torch.use_deterministic_algorithms(False)

    def test_index_put_size1_stride_accumulate(self):
        torch.use_deterministic_algorithms(True)
        try:
            self._check(accumulate=True)
        finally:
            torch.use_deterministic_algorithms(False)


if __name__ == "__main__":
    run_tests()
