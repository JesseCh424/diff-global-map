"""Lightweight plugin package for global_diffusion_map.

Exposes thin wrappers around MMDetection3D training/testing to allow
local customization without modifying upstream entrypoints.
"""

try:  # optional re-exports
    from .core.apis import custom_train_model  # noqa: F401
except Exception:
    pass

