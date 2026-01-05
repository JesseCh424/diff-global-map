"""Test helpers.

Provide a custom multi-GPU test wrapper; falls back to mmdet APIs if present.
"""

from __future__ import annotations

from typing import Any, List


def custom_multi_gpu_test(model, data_loader, tmpdir: str = None, gpu_collect: bool = False) -> List[Any]:
    try:
        from mmdet.apis import multi_gpu_test
        return multi_gpu_test(model, data_loader, tmpdir=tmpdir, gpu_collect=gpu_collect)
    except Exception as e:  # pragma: no cover
        # Fallback: run single_gpu_test across the loader (not efficient, placeholder)
        from mmdet3d.apis import single_gpu_test
        return single_gpu_test(model, data_loader, show=False)

