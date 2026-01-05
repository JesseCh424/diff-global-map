#!/usr/bin/env python
import os
import sys
import runpy

def main():
    # Prepend shim path so that 'projects.mmdet3d_plugin' resolves to our shim first.
    shim_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plugin', 'shims'))
    if shim_root not in sys.path:
        sys.path.insert(0, shim_root)
    # Ensure poly-diffuse package root is on sys.path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    poly_root = os.path.join(repo_root, 'poly-diffuse')
    if poly_root not in sys.path:
        sys.path.insert(0, poly_root)
    # Prefer compiled plugin if available; otherwise fall back to shim
    import importlib.util
    # Ensure torch libs are on loader path for compiled ops
    try:
        import torch, pathlib
        torch_lib = os.path.join(pathlib.Path(torch.__file__).parent, 'lib')
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib}:{os.environ.get('LD_LIBRARY_PATH','')}"
    except Exception:
        pass

    use_compiled = False
    try:
        import importlib
        importlib.import_module('GeometricKernelAttention')
        use_compiled = True
    except Exception:
        use_compiled = False

    if use_compiled:
        # Avoid registry conflict if EfficientNet already registered
        try:
            from mmdet.models import BACKBONES  # type: ignore
            BACKBONES.module_dict.pop('EfficientNet', None)
        except Exception:
            pass
        plugin_init = os.path.abspath(os.path.join(repo_root, 'poly-diffuse', 'projects', 'mmdet3d_plugin', '__init__.py'))
    else:
        plugin_init = os.path.abspath(os.path.join(shim_root, 'projects', 'mmdet3d_plugin', '__init__.py'))

    spec = importlib.util.spec_from_file_location(
        'projects.mmdet3d_plugin', plugin_init,
        submodule_search_locations=[os.path.dirname(plugin_init)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['projects.mmdet3d_plugin'] = module
    spec.loader.exec_module(module)  # type: ignore

    # Patch PolyMetaModel to use stats-driven max_poly/num_vert.
    try:
        stats_path = os.environ.get('AV2_STATS_JSON', os.path.join(os.path.dirname(__file__), '..', 'work_dirs', 'av2_stats.json'))
        import json
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            _max_poly = int(stats.get('num_queries', 50))
            _num_vert = int(stats.get('M', 20))
        else:
            _max_poly, _num_vert = 50, 20
        # Optional overrides for guide memory safety
        _max_poly = int(os.environ.get('AV2_NUM_QUERIES_OVERRIDE', _max_poly))
        _num_vert = int(os.environ.get('AV2_M_OVERRIDE', _num_vert))
        from src.models.polygon_models import polygon_meta as _pm
        _orig_init = _pm.PolyMetaModel.__init__
        def _patched_init(self, input_dim, embed_dim, max_poly=50, num_vert=20):
            if max_poly == 50:
                max_poly = _max_poly
            if num_vert == 20:
                num_vert = _num_vert
            return _orig_init(self, input_dim, embed_dim, max_poly=max_poly, num_vert=num_vert)
        _pm.PolyMetaModel.__init__ = _patched_init  # type: ignore
    except Exception as e:
        print('Warning: failed to monkey-patch PolyMetaModel:', e)

    # Prepare packed proposals（可选）。默认关闭，仅当设置 PREPARE_PACKED_PROPOSALS=1 时启用。
    if os.environ.get('PREPARE_PACKED_PROPOSALS', ''):
        try:
            from global_diffusion_map.tools.prepare_packed_proposals import prepare as _prepare_pack  # type: ignore
            agg_pred_dir = os.environ.get('AGG_PRED_DIR_TRAIN', 'maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train')
            bounds_dir   = os.environ.get('BOUNDS_DIR_TRAIN', agg_pred_dir)
            out_root     = os.environ.get('PACKED_PROPOSALS_TRAIN', 'global_diffusion_map/work_dirs/packed_proposals/av2_oldsplit/train')
            guide_ckpt   = os.environ.get('GUIDE_CKPT', 'global_diffusion_map/ckpts/guide/network-snapshot_m30q64.pth')
            stats_json   = os.environ.get('AV2_STATS_JSON', os.path.join(os.path.dirname(__file__), '..', 'work_dirs', 'av2_stats.json'))
            cfg_path     = os.environ.get('PACK_CFG', 'global_diffusion_map/plugin/configs/global_diffusion/av2_polydiffuse_official_base.py')
            if os.path.isdir(agg_pred_dir):
                _prepare_pack(agg_pred_dir, bounds_dir, out_root, cfg_path, guide_ckpt, stats_json, mode='train')
            agg_pred_val = os.environ.get('AGG_PRED_DIR_VAL', 'maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid')
            bounds_val   = os.environ.get('BOUNDS_DIR_VAL', agg_pred_val)
            out_root_val = os.environ.get('PACKED_PROPOSALS_VAL', 'global_diffusion_map/work_dirs/packed_proposals/av2_oldsplit/val')
            if os.path.isdir(agg_pred_val):
                _prepare_pack(agg_pred_val, bounds_val, out_root_val, cfg_path, guide_ckpt, stats_json, mode='train')
        except Exception as e:
            print('[warn] prepare_packed_proposals failed or skipped:', e)

    # Execute poly-diffuse/train.py with forwarded argv
    train_path = os.path.join(poly_root, 'train.py')
    sys.argv = [train_path] + sys.argv[1:]
    runpy.run_path(train_path, run_name='__main__')

if __name__ == '__main__':
    main()
