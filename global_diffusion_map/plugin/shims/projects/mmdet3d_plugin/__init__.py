import os
import sys
import importlib.util

# Avoid registry name conflict for EfficientNet
try:
    from mmdet.models import BACKBONES
    BACKBONES.module_dict.pop('EfficientNet', None)
except Exception:
    pass

"""Shim loader for PolyDiffuse plugin.
- Removes EfficientNet registry conflicts
- Pre-injects a pure-PyTorch fallback for geometric kernel attention
- Imports the real plugin afterwards so all other modules work as-is
"""

# Pre-inject our fallback geometric kernel attention Function so that
# downstream relative imports resolve to it instead of the compiled op.
_shim_func = os.path.join(os.path.dirname(__file__),
                          'maptr', 'modules', 'ops', 'geometric_kernel_attn',
                          'function', 'geometric_kernel_attn_func.py')
spec_shim = importlib.util.spec_from_file_location(
    'projects.mmdet3d_plugin.maptr.modules.ops.geometric_kernel_attn.function.geometric_kernel_attn_func',
    _shim_func,
    submodule_search_locations=[os.path.dirname(_shim_func)])
mod_shim = importlib.util.module_from_spec(spec_shim)
sys.modules['projects.mmdet3d_plugin.maptr.modules.ops.geometric_kernel_attn.function.geometric_kernel_attn_func'] = mod_shim
spec_shim.loader.exec_module(mod_shim)  # type: ignore

# Also inject parent packages for safety
pkg_function = os.path.dirname(_shim_func)
spec_parent = importlib.util.spec_from_file_location(
    'projects.mmdet3d_plugin.maptr.modules.ops.geometric_kernel_attn.function',
    os.path.join(pkg_function, '__init__.py'),
    submodule_search_locations=[pkg_function])
mod_parent = importlib.util.module_from_spec(spec_parent)
sys.modules['projects.mmdet3d_plugin.maptr.modules.ops.geometric_kernel_attn.function'] = mod_parent
spec_parent.loader.exec_module(mod_parent)  # type: ignore

pkg_ops = os.path.dirname(pkg_function)
spec_ops = importlib.util.spec_from_file_location(
    'projects.mmdet3d_plugin.maptr.modules.ops.geometric_kernel_attn',
    os.path.join(pkg_ops, '__init__.py'),
    submodule_search_locations=[pkg_ops])
mod_ops = importlib.util.module_from_spec(spec_ops)
sys.modules['projects.mmdet3d_plugin.maptr.modules.ops.geometric_kernel_attn'] = mod_ops
spec_ops.loader.exec_module(mod_ops)  # type: ignore

# Load the real plugin package from poly-diffuse folder
_this_dir = os.path.dirname(__file__)
_root = os.path.abspath(os.path.join(_this_dir, '../../../../..'))
_real_init = os.path.join(_root, 'poly-diffuse', 'projects', 'mmdet3d_plugin', '__init__.py')

spec = importlib.util.spec_from_file_location(
    'projects.mmdet3d_plugin', _real_init,
    submodule_search_locations=[os.path.dirname(_real_init)])
module = importlib.util.module_from_spec(spec)
sys.modules['projects.mmdet3d_plugin'] = module
spec.loader.exec_module(module)  # type: ignore
