# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
#
# Regression test for https://github.com/intel/torch-xpu-ops/issues/6321
# (port of the pytorch#203892 test test_split_with_sizes_copy_out_non_canonical_layouts).

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestXpuSplitWithSizesCopyNonCanonical(TestCase):
    def _check(self, x, split_sizes, dim, out):
        torch.split_with_sizes_copy(x, split_sizes, dim=dim, out=out)
        for t, expect in zip(out, x.split_with_sizes(split_sizes, dim=dim)):
            self.assertEqual(t, expect)

    def test_split_with_sizes_copy_out_non_canonical_layouts(self, device):
        # Dense but not contiguous input
        x = torch.arange(12.0, device=device).reshape(3, 4).t()
        self._check(
            x, [2, 2], 0, [torch.empty(2, 3, device=device) for _ in range(2)]
        )
        # Contiguous input whose size-1 split dim has an arbitrary stride;
        # nothing past the output may be written
        x = torch.zeros(64, device=device).as_strided((2, 1, 2), (2, 16, 1))
        x.copy_(torch.arange(4.0, device=device).view(2, 1, 2))
        guard = torch.full((16,), -1.0, device=device)
        self._check(x, [1], 1, [guard[:4].view(2, 1, 2)])
        self.assertEqual(guard[4:], torch.full((12,), -1.0, device=device))
        # Channels-last dim-0 split into channels-last or contiguous outputs
        x = torch.randn(4, 3, 2, 5, device=device).to(
            memory_format=torch.channels_last
        )
        for memory_format in (torch.channels_last, torch.contiguous_format):
            out = [
                torch.empty(2, 3, 2, 5, device=device, memory_format=memory_format)
                for _ in range(2)
            ]
            self._check(x, [2, 2], 0, out)
        self._check(
            x,
            [4, 0],
            0,
            [torch.empty_like(x), torch.empty(0, 3, 2, 5, device=device)],
        )
        # Permuted leading dims, with outputs permuted the same way
        x = torch.randn(3, 4, 6, device=device).transpose(0, 1)
        self._check(
            x,
            [2, 4],
            2,
            [torch.empty_like(x[:, :, :2]), torch.empty_like(x[:, :, 2:])],
        )
        # The split dim's stride is the trailing size product, though dim 0 is
        # physically inside it and dim 2 outside
        x = torch.arange(40.0, device=device).as_strided((2, 10, 2), (1, 2, 20))
        sizes = [4, 4, 2]
        self._check(
            x,
            sizes,
            1,
            [
                torch.empty_strided((2, s, 2), (1, 2, 2 * s), device=device)
                for s in sizes
            ],
        )
        # A single non-empty split laid out like the transposed input
        x = torch.arange(12.0, device=device).reshape(3, 4).t()
        self._check(
            x, [0, 3], 1, [torch.empty(4, 0, device=device), torch.empty_like(x)]
        )


instantiate_device_type_tests(
    TestXpuSplitWithSizesCopyNonCanonical, globals(), only_for="xpu", allow_xpu=True
)
