# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
#
# Regression tests for
#   https://github.com/intel/torch-xpu-ops/issues/6318
#   https://github.com/intel/torch-xpu-ops/issues/6319
# upsample_bilinear2d_backward must run natively (no
# "does not have a deterministic implementation" alert) and bit-reproducibly
# under torch.use_deterministic_algorithms(True), matching CUDA behavior.
#
# NOTE: the F.interpolate python layer routes non-CUDA devices to the
# _upsample_linear_vec decomposition while deterministic algorithms are
# enabled (torch/nn/functional.py, gated in pytorch core). The tests below
# therefore build the forward graph either outside the guard (like
# TestNNDeviceTypeCUDA::test_upsamplingBiMode2d_deterministic) or through the
# native entry point torch._C._nn.upsample_bilinear2d (equivalent to
# TestTorchDeviceTypeCUDA::test_deterministic_interpolate_bilinear mocking
# the decomposition), so the native XPU deterministic kernel is exercised.

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    run_tests,
    TestCase,
)


def _bilinear_native(input, size, align_corners=False, scale_factors=None):
    # Native entry point used by F.interpolate for bilinear mode; skips the
    # python-level deterministic decomposition fallback.
    return torch._C._nn.upsample_bilinear2d(
        input, size, align_corners, scale_factors
    )


class TestXpuUpsampleBilinear2dBackwardDeterministic(TestCase):
    # Mirrors TestNNDeviceTypeCUDA::test_upsamplingBiMode2d_deterministic: the
    # forward graph is built outside the deterministic guard, so the backward
    # must be served by the native deterministic kernel.
    @dtypes(torch.float32, torch.float64)
    def test_backward_native_deterministic(self, device, dtype):
        for scale_factor in [0.5, 1.5, 2]:
            for align_corners in [False, True]:
                for memory_format in (
                    torch.contiguous_format,
                    torch.channels_last,
                ):
                    in_t = (
                        torch.ones(2, 3, 8, 8, device=device, dtype=dtype)
                        .contiguous(memory_format=memory_format)
                        .requires_grad_()
                    )
                    out_t = F.interpolate(
                        in_t,
                        scale_factor=scale_factor,
                        mode="bilinear",
                        align_corners=align_corners,
                    )
                    grad_out = torch.randn_like(out_t).contiguous(
                        memory_format=memory_format
                    )
                    with DeterministicGuard(True):
                        out_t.backward(grad_out)
                    self.assertTrue(
                        in_t.grad.is_contiguous(memory_format=memory_format)
                    )

                    input_ref = in_t.detach().cpu().requires_grad_()
                    output_ref = F.interpolate(
                        input_ref,
                        scale_factor=scale_factor,
                        mode="bilinear",
                        align_corners=align_corners,
                    )
                    output_ref.backward(grad_out.cpu())
                    atol, rtol = (
                        (1e-10, 1e-10)
                        if dtype == torch.float64
                        else (1e-5, 1e-5)
                    )
                    self.assertEqual(
                        in_t.grad.cpu(), input_ref.grad, atol=atol, rtol=rtol
                    )

    # Mirrors TestTorchDeviceTypeCUDA::test_deterministic_interpolate_bilinear:
    # two backward runs must agree bit-exactly and match the CPU reference.
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_interpolate_bilinear_native_bit_reproducible(self, device, dtype):
        atol, rtol = {
            torch.float16: (1e-2, 1e-2),
            torch.bfloat16: (5e-2, 5e-2),
            torch.float32: (1e-5, 1e-5),
            torch.float64: (1e-10, 1e-10),
        }[dtype]
        for shape, size, channels_last in [
            ((1, 8, 4, 5), (8, 10), True),
            ((2, 3, 4, 5), (8, 10), False),
        ]:
            memory_format = (
                torch.channels_last if channels_last else torch.contiguous_format
            )
            n, c = shape[0], shape[1]
            input = (
                torch.randn(*shape, device=device, dtype=dtype)
                .contiguous(memory_format=memory_format)
                .requires_grad_()
            )
            output_grad = (
                torch.randn(n, c, *size, device=device, dtype=dtype)
                .contiguous(memory_format=memory_format)
            )

            grad = None
            with DeterministicGuard(True):
                for _ in range(2):
                    input.grad = None
                    output = _bilinear_native(input, size)
                    output.backward(output_grad)
                    if grad is None:
                        grad = input.grad.detach().clone()
                    else:
                        self.assertEqual(grad, input.grad, atol=0, rtol=0)
            if channels_last:
                self.assertTrue(
                    grad.is_contiguous(memory_format=torch.channels_last)
                )

            reference_input = input.detach().cpu().contiguous().requires_grad_()
            reference_output = F.interpolate(
                reference_input, size=size, mode="bilinear", align_corners=False
            )
            reference_output.backward(output_grad.cpu().contiguous())
            self.assertEqual(
                grad.cpu(), reference_input.grad, atol=atol, rtol=rtol
            )

    # scale_factor + recompute_scale_factor variant from
    # test_deterministic_interpolate_bilinear (output size precomputed on the
    # host, as recompute_scale_factor does).
    def test_interpolate_bilinear_scale_factor_recompute(self, device):
        shape, scale_factor = (1, 3, 4, 5), (1.7, 2.0)
        _, _, h, w = shape
        sh, sw = scale_factor
        size = (int(h * sh), int(w * sw))
        input = (
            torch.randn(*shape, device=device, dtype=torch.float32)
            .contiguous()
            .requires_grad_()
        )
        output_grad = torch.randn(
            1, 3, *size, device=device, dtype=torch.float32
        )

        grad = None
        with DeterministicGuard(True):
            for _ in range(2):
                input.grad = None
                output = _bilinear_native(input, size)
                output.backward(output_grad)
                if grad is None:
                    grad = input.grad.detach().clone()
                else:
                    self.assertEqual(grad, input.grad, atol=0, rtol=0)

        reference_input = input.detach().cpu().contiguous().requires_grad_()
        reference_output = F.interpolate(
            reference_input,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True,
        )
        reference_output.backward(output_grad.cpu().contiguous())
        self.assertEqual(
            grad.cpu(), reference_input.grad, atol=1e-5, rtol=1e-5
        )


instantiate_device_type_tests(
    TestXpuUpsampleBilinear2dBackwardDeterministic,
    globals(),
    only_for="xpu",
    allow_xpu=True,
)
