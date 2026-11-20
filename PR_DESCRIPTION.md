# Sync test_str_storage expectations with upstream storage repr change (pytorch#204532)

Fixes #6298

## Summary

Upstream pytorch 2c3e40c2 ("Print only metadata in Storage repr() and str(),
never the data", pytorch#204532, merged 2026-11-19) changed `str(TypedStorage)`
to print only the `[torch.storage.TypedStorage(...)]` header and updated the
expectations in `test/test_fake_tensor.py` accordingly.

`test/xpu/test_fake_tensor_xpu.py` still carries the pre-2c3e40c2 expectations
(values + header), so once the CI's pytorch nightly picked up the change,
`FakeTensorOperatorInvariants.test_str_storage` (and its
`make_propagate_real_tensors_cls` variant `test_str_storage_propagate_real_tensors`)
started failing `assertExpectedInline` — actual output is header-only while the
test still expects the data lines.

This PR ports the test-side half of 2c3e40c2: replace the three expected
strings in `test_str_storage` with the header-only form, exactly matching
upstream `test/test_fake_tensor.py`.

## Testing

- A770 (DG2), torch 2.16.0.dev nightly (post-2c3e40c2):
  - Before this PR: `test_str_storage` FAIL (`' 0.0
 0.0
 0.0
[...]' != '[...]'`)
  - After this PR: `test_str_storage` PASS (and the propagate_real_tensors
    variant, generated from the same test body)
- Note: on torch builds older than 2c3e40c2 the updated test fails by design —
  the expectations intentionally track current upstream main behavior, same as
  the rest of this test file which is periodically synchronized from upstream.
