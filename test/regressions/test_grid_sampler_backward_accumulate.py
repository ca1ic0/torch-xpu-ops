# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase


class TestGridSamplerBackwardAccumulate(TestCase):
    def _run(self, dtype):
        # 2048 channels each contribute exactly 1 to the x-gradient at grid
        # coordinate (0, 0); the exact total 2048 is representable in float16
        # and bfloat16. The backward kernel used to keep the accumulator in
        # the input dtype and stalled at 1024 (fp16) / 128 (bf16).
        C = 2048
        inp = torch.zeros(1, C, 2, 2, device="xpu", dtype=dtype)
        inp[..., 1] = 1.0
        inp.requires_grad_(True)
        grid = torch.zeros(1, 1, 1, 2, device="xpu", dtype=dtype, requires_grad=True)
        out = F.grid_sample(inp, grid, align_corners=False)
        out.sum().backward()
        return grid.grad[0, 0, 0].tolist()

    def test_grid_sampler_2d_backward_accumulate(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            gx, gy = self._run(dtype)
            self.assertEqual(
                (gx, gy),
                (2048.0, 0.0),
                msg=f"grid gradient for {dtype} accumulated in reduced precision",
            )


if __name__ == "__main__":
    run_tests()
