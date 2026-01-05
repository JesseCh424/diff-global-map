#!/usr/bin/env python
"""
Quick sanity evaluator for ProposalEncoder effect.

Compares denoiser loss with and without proposal vector-condition on a few batches.

Usage:
  python -u global_diffusion_map/tools/eval_proposal_encoder.py \
    --config global_diffusion_map/plugin/configs/global_diffusion/av2_polydiffuse_official_base.py \
    --ckpt <denoise_network_snapshot.pth> \
    --guide-ckpt global_diffusion_map/ckpts/guide/network-snapshot_m30q64.pth \
    --stats-json global_diffusion_map/work_dirs/av2_stats.json \
    --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
    --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
    --proposal-root global_diffusion_map/work_dirs/packed_proposals/av2_oldsplit/val \
    --scene-list <optional text file> \
    --batches 4
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import torch
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset


def _patch_paths():
    # Ensure plugin and poly-diffuse are importable (mirror run_train.py)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    poly_root = os.path.join(repo_root, 'poly-diffuse')
    if poly_root not in sys.path:
        sys.path.insert(0, poly_root)
    # Prefer compiled plugin if available; otherwise fall back to shim
    import importlib.util
    shim_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plugin', 'shims'))
    use_compiled = False
    try:
        import importlib
        importlib.import_module('GeometricKernelAttention')
        use_compiled = True
    except Exception:
        use_compiled = False
    if use_compiled:
        plugin_init = os.path.abspath(os.path.join(poly_root, 'projects', 'mmdet3d_plugin', '__init__.py'))
    else:
        plugin_init = os.path.abspath(os.path.join(shim_root, 'projects', 'mmdet3d_plugin', '__init__.py'))
    spec = importlib.util.spec_from_file_location(
        'projects.mmdet3d_plugin', plugin_init,
        submodule_search_locations=[os.path.dirname(plugin_init)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['projects.mmdet3d_plugin'] = module
    spec.loader.exec_module(module)  # type: ignore


def load_stats(stats_json: str) -> tuple[int, int]:
    if os.path.exists(stats_json):
        with open(stats_json, 'r') as f:
            data = json.load(f)
        return int(data.get('M', 20)), int(data.get('num_queries', 50))
    return 20, 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--guide-ckpt', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--proposal-root', required=True)
    ap.add_argument('--scene-list', default=None)
    ap.add_argument('--batches', type=int, default=2)
    args = ap.parse_args()

    _patch_paths()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load stats (caps)
    M, num_queries = load_stats(args.stats_json)

    # Build dataset (val mode)
    from global_diffusion_map.plugin.datasets.av2_diffusion_dataset import AV2GlobalDiffusionDataset
    ds = AV2GlobalDiffusionDataset(
        static_root=args.static_root,
        rendered_gt_root=args.rendered_root,
        semantic_root=None,
        scene_list=args.scene_list,
        use_condition='10',
        M=M,
        num_queries=num_queries,
        drop_instance=False,
        load_image=True,
        cond_max_side=1024,
        cond_fixed_size=(1024, 1024),
        proposal_root=args.proposal_root,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    # Build guide
    from src.models.polygon_models.polygon_meta import PolyMetaModel  # type: ignore
    net_guide = PolyMetaModel(input_dim=128, embed_dim=256, max_poly=num_queries, num_vert=M).to(device)
    gsd = torch.load(args.guide_ckpt, map_location='cpu')
    key = 'net'
    state = gsd.get(key, gsd)
    try:
        net_guide.load_state_dict(state)
    except Exception:
        net_guide.load_state_dict(state, strict=False)
    net_guide.eval()

    # Build denoiser model
    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')).to(device)
    precond = None
    # Wrap with EDMPrecond; reuse the existing EDMPrecond from poly-diffuse
    from src.models.networks import EDMPrecond
    precond = EDMPrecond(model_type='maptr', model=model, sigma_min=0.002, sigma_max=2.0).to(device)
    # Load denoise weights
    dsd = torch.load(args.ckpt, map_location='cpu')
    net_state = dsd.get('net', dsd)
    try:
        model.load_state_dict(net_state)
    except Exception:
        model.load_state_dict(net_state, strict=False)
    precond.eval()

    # Loss
    from src.losses.loss_denoise import EDMLoss as _Loss
    loss_fn = _Loss(P_mean=-0.5, P_std=1.5, sigma_data=1.0, pc_range=[-1,-1,-2,1,1,2], lambda_dir=1e-3)

    # Evaluate a few batches
    import numpy as np
    deltas = []
    with torch.no_grad():
        it = iter(loader)
        for _ in range(args.batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            images = batch['gt_bboxes_3d'].to(device)
            attn_mask = batch['pts_mask'].to(device)
            mk = {
                'img': batch['img'].to(device),
                'poly_class': batch['gt_labels_3d'].to(device),
                'poly_mask': attn_mask,
                'img_metas': batch['img_metas']
            }
            # Build mu_guide
            mu_c, sig = net_guide(images, attn_mask, mk['poly_class'])
            mu = mu_c[:, :, None, :].repeat(1, 1, images.shape[2], 1)
            # With proposals
            if 'proposal_pts' in batch:
                mk_prop = dict(mk)
                mk_prop['proposal_pts'] = batch['proposal_pts'].to(device)
                mk_prop['proposal_mask'] = batch['proposal_mask'].to(device)
                if 'proposal_labels' in batch:
                    mk_prop['proposal_labels'] = batch['proposal_labels'].to(device)
            else:
                mk_prop = mk
            # Loss with proposals
            l_all = loss_fn(precond, net_guide, images, z_init=None, **mk_prop)
            last = float(l_all[1].mean().item())
            # Loss without proposals
            l_np = loss_fn(precond, net_guide, images, z_init=None, **mk)
            last_np = float(l_np[1].mean().item())
            deltas.append((last_np, last))
    if len(deltas) == 0:
        print('[error] no batches evaluated')
        return
    base = np.mean([a for a, b in deltas])
    with_prop = np.mean([b for a, b in deltas])
    imp = base - with_prop
    print(f'[eval] mean last-layer loss no-prop={base:.4f}, with-prop={with_prop:.4f}, delta={imp:.4f}')


if __name__ == '__main__':
    main()

