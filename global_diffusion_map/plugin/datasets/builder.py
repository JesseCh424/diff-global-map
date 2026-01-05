"""Dataset/dataloader builders (thin wrappers).

This allows us to extend behavior (samplers, sharding, etc.) while
defaulting to the upstream implementation when possible.
"""

from __future__ import annotations

from typing import Any


def build_dataloader(
    dataset: Any,
    samples_per_gpu: int,
    workers_per_gpu: int,
    num_gpus: int = 1,
    dist: bool = False,
    shuffle: bool = True,
    seed: int = None,
    shuffler_sampler: Any = None,
    nonshuffler_sampler: Any = None,
    **kwargs,
):
    """Forward to mmdet3d.datasets.build_dataloader, ignoring extra kwargs."""
    from mmdet3d.datasets import build_dataloader as _build

    return _build(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        num_gpus=num_gpus,
        dist=dist,
        shuffle=shuffle,
        seed=seed,
        **{k: v for k, v in kwargs.items() if k not in {'shuffler_sampler', 'nonshuffler_sampler'}},
    )

