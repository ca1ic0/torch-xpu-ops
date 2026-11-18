# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression tests for the deterministic CTC loss backward on XPU.

Upstream pytorch/pytorch@56ac168 (#201617) made the CUDA native CTC loss
backward deterministic: under ``torch.use_deterministic_algorithms(True)``
the kernel skips the atomicAdd-based large-alphabet path and uses the naive
per-element gradient algorithm, and the unconditional
``alertNotDeterministic`` call was dropped.

The XPU kernel kept the old structure and always raised under deterministic
mode.  These tests mirror the upstream oracle ``test_CTCLoss_deterministic``
(test/test_nn.py): no alert, loss bitwise equal to the nondeterministic
reference, gradient within tolerance, repeated runs bitwise identical, and
no warning under ``warn_only=True``.
"""

import unittest
import warnings

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase

CASES = {
    "small_path": ((40, 5, 4), [5, 8, 16, 12, 12], [40] * 5, False, False, False),
    "long_targets_large_alphabet": (
        (300, 5, 257), [127] * 5, [300, 300, 300, 300, 297], False, False, True),
    "large_alphabet": (
        (240, 5, 1000), [5, 8, 16, 12, 12], [240, 240, 240, 240, 237], True, False, False),
    "boundaries_zero_infinity": (
        (240, 6, 36), [0, 0, 64, 0, 260, 1], [240, 120, 200, 0, 240, 0], True, True, False),
}


class TestCTCLossDeterministic(TestCase):
    def _inputs(self, dtype, device, case):
        shape, target_lengths, input_lengths, concatenated, zero_infinity, repeated = case
        time, batch, classes = shape
        torch.manual_seed(0)
        log_probs = torch.randn(batch, time, classes, device=device, dtype=dtype)
        log_probs = log_probs.log_softmax(2).transpose(0, 1).detach().requires_grad_()
        targets = torch.randint(
            0, classes - 1, (batch, 2 * max(target_lengths)), device=device, dtype=torch.long
        )[:, ::2]
        targets.add_(targets >= 3)
        if repeated:
            split = max(target_lengths) // 2
            targets[:, :split] = 0
            targets[:, split:] = 1
        if concatenated:
            targets = torch.cat([t[:n] for t, n in zip(targets, target_lengths)]).int()
        grad_out = torch.randn(batch * 2, device=device, dtype=dtype)[::2]
        return log_probs, targets, input_lengths, target_lengths, zero_infinity, grad_out

    def _loss_and_grad(self, log_probs, targets, il, tl, zi, grad_out):
        loss = F.ctc_loss(
            log_probs, targets, il, tl, blank=3, reduction="none", zero_infinity=zi
        )
        (grad,) = torch.autograd.grad(loss, log_probs, grad_out)
        return loss, grad

    def test_ctc_loss_backward_deterministic_float(self, device):
        self._run_oracle(torch.float32, 1e-3, device)

    @unittest.skipIf(
        not torch.xpu.is_available()
        or not getattr(torch.xpu.get_device_properties(0), "has_fp64", True),
        "fp64 unsupported on this XPU device",
    )
    def test_ctc_loss_backward_deterministic_double(self, device):
        self._run_oracle(torch.float64, 1e-11, device)

    def _run_oracle(self, dtype, atol, device):
        for name, case in CASES.items():
            lp, tg, il, tl, zi, go = self._inputs(dtype, device, case)
            torch.use_deterministic_algorithms(False)
            ref_loss, ref_grad = self._loss_and_grad(lp, tg, il, tl, zi, go)

            torch.use_deterministic_algorithms(True)
            try:
                loss, grad = self._loss_and_grad(lp, tg, il, tl, zi, go)
                reps = [self._loss_and_grad(lp, tg, il, tl, zi, go) for _ in range(3)]
            finally:
                torch.use_deterministic_algorithms(False)

            self.assertTrue(torch.equal(loss.cpu(), ref_loss.cpu()), name)
            self.assertTrue(
                torch.allclose(grad.cpu(), ref_grad.cpu(), atol=atol, rtol=0), name
            )
            for rep_loss, rep_grad in reps:
                self.assertTrue(torch.equal(rep_loss.cpu(), loss.cpu()), name)
                self.assertTrue(torch.equal(rep_grad.cpu(), grad.cpu()), name)

            torch.use_deterministic_algorithms(True, warn_only=True)
            try:
                with warnings.catch_warnings(record=True) as rec:
                    warnings.simplefilter("always")
                    self._loss_and_grad(lp, tg, il, tl, zi, go)
                nondet = [w for w in rec if "deterministic" in str(w.message).lower()]
                self.assertEqual(len(nondet), 0, name)
            finally:
                torch.use_deterministic_algorithms(False)


instantiate_device_type_tests(
    TestCTCLossDeterministic, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
