# [indexing] Step index_put_ batches with the dense stride, not src.stride(i-1)

**Fixes #6084**

## Summary

`computeLinearIndex` (`src/ATen/native/xpu/sycl/IndexingUtils.h`) derived
`strideBefore` from `src.stride(i - 1)` of the dim right before the first
indexed dim. A **contiguous** tensor can carry an arbitrary stride on a
size-1 dim (e.g. an `as_strided` view), and the sort-based `index_put_`
kernels address `self` as dense (`bi_ * stride_before_`,
`src/ATen/native/xpu/sycl/Indexing.h:838`), so writes stepped between batches
with the bogus stride and landed outside the indexed tensor:

```python
base = torch.zeros(16, device="xpu")
x = base.as_strided((2, 1, 2), (2, 8, 1))   # x.is_contiguous() is True
torch.use_deterministic_algorithms(True)
x[:, :, torch.tensor([1], device="xpu")] = torch.ones(2, 1, 1, device="xpu")
# expected base offsets 1, 3 written; XPU wrote 1 and 9 (stride(1) == 8)
```

This is the XPU twin of the CUDA bug fixed on pytorch main
(pytorch/pytorch#203889): use the contiguous stride `strides[i-1]` from
`computeLinearStride`, with the same comment. Both the accumulate and
non-accumulate deterministic paths are affected.

## Testing

- New regression test `test/regressions/test_index_put_size1_stride.py`
  (CPU + XPU, accumulate and non-accumulate, deterministic mode): XPU case
  wrote to offsets `{1, 9}` before, `{1, 3}` after; CPU always correct.
