# Shim to load PolyDiffuse plugin without registry conflicts.
try:
    from mmdet.models import BACKBONES
    BACKBONES.module_dict.pop('EfficientNet', None)
except Exception:
    pass

# Import the actual plugin (registers MapTR & related modules)
import projects.mmdet3d_plugin  # noqa: F401
