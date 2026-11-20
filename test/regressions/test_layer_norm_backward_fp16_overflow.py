# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Owner(s): ["module: intel"]

# Regression test for https://github.com/intel/torch-xpu-ops/issues/6310.
#
# layer_norm backward used to evaluate `fH * gamma * dY` in float16 (fH is int,
# at::Half arithmetic overloads convert it to Half first). For normalized dim
# N >= 8192 the product overflows fp16 range and grad_input becomes inf,
# while CPU/CUDA accumulate in the acc type. Verify grad_input stays finite
# and matches a float32 CPU reference for the large-N shapes.

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestLayerNormBackwardFp16Overflow(TestCase):
    def _backward(self, fn, device, dtype, args, grad):
        inputs = [t.to(device, dtype).requires_grad_() for t in args]
        out = fn(*inputs)
        return torch.autograd.grad(out, inputs, grad.to(device, dtype))[0]

    def test_layer_norm_backward_fp16_large_n(self, device):
        # N >= 8192 is where the fp16 overflow starts (see the issue table).
        for n in (8192, 20001, 65536):
            shape, nshape = (2, n), (n,)
            torch.manual_seed(0)
            args = [
                torch.randn(s, dtype=torch.half)
                for s in (shape, nshape, nshape)
            ]
            grad = torch.randn(shape, dtype=torch.half)

            actual = self._backward(
                lambda x, w, b: F.layer_norm(x, nshape, w, b),
                device,
                torch.half,
                args,
                grad,
            )
            expected = self._backward(
                lambda x, w, b: F.layer_norm(x, nshape, w, b),
                "cpu",
                torch.float,
                args,
                grad,
            )

            self.assertFalse(
                torch.isinf(actual.cpu().float()).any(),
                f"grad_input contains inf for N={n}",
            )
            # Same tolerance as the upstream test_layer_norm_backward_shapes.
            self.assertEqual(
                actual.cpu().float(),
                expected,
                atol=2e-3 * expected.abs().max().item(),
                rtol=0,
            )

    def test_rms_norm_backward_fp16_large_n(self, device):
        # rms_norm backward shares LayerNormBackward::update() and hits the
        # same `fH * gamma * dY` expression.
        n = 20001
        torch.manual_seed(0)
        args = [torch.randn(s, dtype=torch.half) for s in ((2, n), (n,))]
        grad = torch.randn((2, n), dtype=torch.half)

        actual = self._backward(
            lambda x, w: F.rms_norm(x, (n,), w),
            device,
            torch.half,
            args,
            grad,
        )
        expected = self._backward(
            lambda x, w: F.rms_norm(x, (n,), w),
            "cpu",
            torch.float,
            args,
            grad,
        )

        self.assertFalse(
            torch.isinf(actual.cpu().float()).any(),
            "rms_norm grad_input contains inf",
        )
        self.assertEqual(
            actual.cpu().float(),
            expected,
            atol=2e-2 * max(expected.abs().max().item(), 1.0),
            rtol=0,
        )


instantiate_device_type_tests(
    TestLayerNormBackwardFp16Overflow, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
