"""Custom API wrappers.

Currently forwards to mmdet3d.apis.train_model, leaving space to extend
logging, hooks, or special behaviors without touching upstream code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from mmdet3d.apis import train_model as _train_model
except Exception as e:  # pragma: no cover
    raise


def custom_train_model(
    model,
    datasets: List[Any],
    cfg,
    distributed: bool = False,
    validate: bool = False,
    timestamp: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
):
    """Thin wrapper around mmdet3d.apis.train_model.

    This is a hook for downstream customization.
    """
    return _train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        meta=meta,
    )

