# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase

# dispatch_to_loop_scan_kernel (ScanUtils.h) selects the loop-scan kernel when
# scanning the contiguous last dim (stride == 1) with batch > 128 and
# problem < 16384. The loop-scan kernel backs cumsum, cumprod and
# logcumsumexp; cummax/cummin go through the with-indices kernel, which shares
# LoopScanConfig launch geometry. Shapes below cover both sides of the
# batch/problem thresholds, the grid-stride loop (batch * sg_size larger than
# the device's resident work items), and a single-batch corner. Scanning any
# dim other than the last is strided (stride > 1) and exercises the
# segment-scan fallback instead.
_shapes = [
    (256, 1000),  # loop scan: batch and problem both in range
    (1000, 512),  # loop scan
    (200, 8192),  # loop scan: multiple problem chunks per row
    (129, 33),  # loop scan: just above the batch threshold
    (200, 5),  # loop scan: problem < sub-group width, single partial chunk
    (8, 64, 300),  # loop scan on dim 2: batch formed by collapsed outer dims
    (65536, 40),  # loop scan: batch * 32 exceeds resident work items,
    # exercising the grid-stride loop
    (64, 4096),  # segment scan: batch <= 128
    (1, 5000),  # segment scan: single batch
]

_float_dtypes = (
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
)
_int_dtypes = (
    torch.int32,
    torch.int64,
)
_complex_dtypes = (
    torch.complex64,
    torch.complex128,
)


class TestScan(TestCase):
    def _input(self, shape, dtype, op):
        if dtype in _int_dtypes:
            # Integer scans are exact (wrap-around included), so CPU and XPU
            # must agree bit for bit — a much stronger oracle than any float
            # tolerance.
            return torch.randint(-3, 4, shape, dtype=dtype)
        x = torch.randn(shape, dtype=dtype)
        if op is torch.cumprod:
            # Keep partial products in a numerically sane range over long rows.
            x = 1 + 0.01 * x
        return x

    def _reference(self, op, x, dim, dtype):
        # float64/complex128 reference, rounded back to the test dtype: a
        # kernel that silently narrows its accumulator still fails, without
        # relaxing tolerances against a same-dtype CPU reference.
        if dtype.is_complex:
            ref_dtype = torch.complex128
        elif dtype.is_floating_point:
            ref_dtype = torch.float64
        else:
            ref_dtype = dtype
        return op(x.to(ref_dtype), dim).to(dtype)

    def _test_scan_op(self, op, dtype):
        device = torch.device(self.device)
        for shape in _shapes:
            x = self._input(shape, dtype, op)
            x_dev = x.to(device)
            for dim in range(x.ndim):
                expected = self._reference(op, x, dim, dtype)
                actual = op(x_dev, dim).cpu()
                self.assertEqual(expected, actual)

    @dtypes(*_float_dtypes, *_int_dtypes, *_complex_dtypes)
    def test_cumsum_loop_scan(self, dtype):
        self._test_scan_op(torch.cumsum, dtype)

    @dtypes(*_float_dtypes, *_int_dtypes, *_complex_dtypes)
    def test_cumprod_loop_scan(self, dtype):
        self._test_scan_op(torch.cumprod, dtype)

    @dtypes(*_float_dtypes, *_complex_dtypes)
    def test_logcumsumexp_loop_scan(self, dtype):
        self._test_scan_op(torch.logcumsumexp, dtype)

    def test_logcumsumexp_special_values(self):
        # logcumsumexp is the only loop-scan caller with a custom functor
        # (_log_add_exp); exercise its -inf/nan branches on loop-scan shapes.
        device = torch.device(self.device)
        for shape in [(256, 1000), (200, 5)]:
            x = torch.randn(shape, dtype=torch.float32)
            x[:, 0] = float("-inf")
            x[::7, -1] = float("inf")
            x[1::13, 1] = float("nan")
            for dim in range(x.ndim):
                expected = torch.logcumsumexp(x, dim)
                actual = torch.logcumsumexp(x.to(device), dim).cpu()
                self.assertEqual(expected, actual, equal_nan=True)

    def test_cumsum_aliased_out(self):
        # UniqueKernels calls scan() with input and output being the same
        # tensor, so alias-safety is a real invariant of the loop-scan kernel.
        device = torch.device(self.device)
        for shape in [(256, 1000), (200, 5)]:
            x = torch.randn(shape, dtype=torch.float32)
            x_dev = x.to(device)
            torch.cumsum(x_dev, 1, out=x_dev)
            expected = torch.cumsum(x.double(), 1).float()
            self.assertEqual(expected, x_dev.cpu())

    def _test_cumminmax(self, op):
        # cummax/cummin use the with-indices kernel, not the loop-scan kernel;
        # kept here because both share LoopScanConfig launch geometry.
        device = torch.device(self.device)
        for shape in _shapes:
            x = torch.randn(shape, dtype=torch.float32)
            x_dev = x.to(device)
            for dim in range(x.ndim):
                ref_val, _ = op(x, dim)
                val, idx = op(x_dev, dim)
                self.assertEqual(ref_val, val.cpu())
                # Indices may tie-break differently across backends; assert the
                # returned indices reproduce the returned values instead.
                self.assertEqual(val.cpu(), x.gather(dim, idx.cpu()))

    def test_cummax_loop_scan(self):
        self._test_cumminmax(torch.cummax)

    def test_cummin_loop_scan(self):
        self._test_cumminmax(torch.cummin)


instantiate_device_type_tests(TestScan, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
