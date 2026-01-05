#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from mmcv import Config

# Guide network (PolyMeta)
from mmdet3d.models import build_model

# Ensure plugin importability and compiled ops path (mirror run_train env setup)
def _prep_env():
    import sys, pathlib
    # Add repo and poly-diffuse roots to sys.path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    poly_root = os.path.join(repo_root, 'poly-diffuse')
    if poly_root not in sys.path:
        sys.path.insert(0, poly_root)
    # LD_LIBRARY_PATH for compiled ops
    try:
        import torch
        torch_lib = os.path.join(pathlib.Path(torch.__file__).parent, 'lib')
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib}:{os.environ.get('LD_LIBRARY_PATH','')}"
    except Exception:
        pass
    # Avoid duplicate EfficientNet registration if plugin imported elsewhere
    try:
        from mmdet.models import BACKBONES  # type: ignore
        BACKBONES.module_dict.pop('EfficientNet', None)
    except Exception:
        pass


def _load_pickle(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _save_pickle(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def _uniform_sample(poly: np.ndarray, M: int) -> np.ndarray:
    from shapely.geometry import LineString
    ls = LineString(poly)
    if ls.length <= 1e-6:
        p = np.array(ls.coords[0], dtype=np.float32)
        return np.tile(p[None, :], (M, 1))
    dists = np.linspace(0.0, ls.length, num=M, dtype=np.float32)
    pts = [list(ls.interpolate(float(d)).coords)[0] for d in dists]
    return np.asarray(pts, dtype=np.float32)


def _normalize_xy(xy: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)
    out = np.empty_like(xy, dtype=np.float32)
    out[:, 0] = ((xy[:, 0] - minx) / w) * 2.0 - 1.0
    out[:, 1] = ((xy[:, 1] - miny) / h) * 2.0 - 1.0
    return out


def _denormalize_xy(xy: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)
    out = np.empty_like(xy, dtype=np.float32)
    out[:, 0] = (xy[:, 0] + 1.0) * 0.5 * w + minx
    out[:, 1] = (xy[:, 1] + 1.0) * 0.5 * h + miny
    return out


 


def _load_raster(path: str, cond_max_side: Optional[int] = 1024, cond_fixed_size: Optional[Tuple[int, int]] = (1024, 1024)) -> np.ndarray:
    """Load conditioning raster and align to training policy.

    - Optional downscale to `cond_max_side` (longer side) using LANCZOS.
    - Optional letterbox to `cond_fixed_size` canvas (top-left paste, black background).
    - Output shape: (1,1,3,H,W) float32 in [0,1].
    """
    img = Image.open(path).convert('RGB')
    w, h = img.size
    # Optional downscale to limit max side first
    if cond_max_side is not None:
        ms = max(w, h)
        if ms > cond_max_side and ms > 0:
            scale = cond_max_side / float(ms)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            try:
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            except Exception:
                img = img.resize((new_w, new_h))
            w, h = img.size
    # Optional letterbox to fixed canvas to guarantee batching
    if cond_fixed_size is not None:
        tgt_h, tgt_w = int(cond_fixed_size[0]), int(cond_fixed_size[1])
        if w == 0 or h == 0:
            canvas = Image.new('RGB', (tgt_w, tgt_h), (0, 0, 0))
            img = canvas
        else:
            # scale to fit inside target while preserving aspect
            scale = min(tgt_w / float(w), tgt_h / float(h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            if (new_w, new_h) != (w, h):
                try:
                    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                except Exception:
                    img_resized = img.resize((new_w, new_h))
            else:
                img_resized = img
            canvas = Image.new('RGB', (tgt_w, tgt_h), (0, 0, 0))
            # Paste top-left for determinism (matches training dataset)
            canvas.paste(img_resized, (0, 0))
            img = canvas
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # C,H,W
    return arr[None, None, ...]  # (len_queue=1, num_cams=1, C,H,W)


# (No curve simplification; keep model output M points to match training)


@torch.no_grad()
def edm_sampler(
    net,
    latents,
    mu_guide,
    model_kwargs,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 0.0,
    S_min: float = 0.0,
    S_max: float = float('inf'),
    S_noise: float = 1.0,
    second_order: bool = True,
    keep_intermediates: bool = False,
):
    device = latents.device
    # Clamp noise levels to network-supported range (align with PolyDiffuse generate.py)
    try:
        sigma_min = max(sigma_min, getattr(net, 'sigma_min', sigma_min))
        sigma_max = min(sigma_max, getattr(net, 'sigma_max', sigma_max))
    except Exception:
        pass
    # Time steps
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    t_steps[-1] = t_steps[-2] * 0.5

    x_next = latents.to(torch.float64)
    intermediates = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        # Increase noise temporarily (EDM churn)
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0  # type: ignore[name-defined]
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_next + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_next)

        # Feature caching policy
        model_kwargs['cache_image_feat'] = (i == 0)
        model_kwargs['use_cached_feat'] = (i > 0)

        # Euler step
        denoised = net(x_hat, t_hat, mu_guide, **model_kwargs).to(torch.float64)
        denoised = denoised[-1]
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Optional 2nd order correction
        if second_order and i < num_steps - 1:
            model_kwargs['cache_image_feat'] = False
            model_kwargs['use_cached_feat'] = True
            den2 = net(x_next, t_next, mu_guide, **model_kwargs).to(torch.float64)
            den2 = den2[-1]
            d_prime = (x_next - den2) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        if keep_intermediates:
            intermediates.append(x_next.detach().clone())

    net.model.clear_cache()
    if keep_intermediates:
        return intermediates
    return x_next


def build_net(cfg_path: str, ckpt_path: str, device: str = 'cuda'):
    _prep_env()
    from src.models.networks import EDMPrecond
    cfg = Config.fromfile(cfg_path)
    if hasattr(cfg, 'plugin') and cfg.plugin:
        assert hasattr(cfg, 'plugin_dir')
        import importlib
        module_path = cfg.plugin_dir.replace('/', '.')[:-1] if cfg.plugin_dir.endswith('/') else cfg.plugin_dir.replace('/', '.')
        importlib.import_module(module_path)
    model = build_model(cfg.model)
    model.init_weights()
    net = EDMPrecond(model_type='maptr', model=model, sigma_data=1.0)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state['net'], strict=False)
    net.to(device).eval()
    return net, cfg


def build_guide(model, guide_ckpt_path: str, device: str = 'cuda', num_verts: int = 20, num_queries: int = 50):
    """Instantiate PolyMetaModel with dims aligned to MapTR head and load ckpt.
    Ensure max_poly (num_queries) matches training caps to avoid shape mismatch.
    """
    # Derive dims from MapTR
    pe_dim = int(getattr(model.pts_bbox_head.positional_encoding, 'num_feats'))
    embed_dim = int(getattr(model.pts_bbox_head.transformer, 'embed_dims'))
    # Lazy import after _prep_env injected poly-diffuse to sys.path
    from src.models.polygon_models.polygon_meta import PolyMetaModel  # type: ignore
    net_guide = PolyMetaModel(input_dim=pe_dim, embed_dim=embed_dim, max_poly=int(num_queries), num_vert=int(num_verts))
    state = torch.load(guide_ckpt_path, map_location='cpu')
    net_guide.load_state_dict(state['net'], strict=False)
    net_guide.to(device).eval()
    return net_guide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Config used to define the model (auto_denoise.py recommended)')
    ap.add_argument('--ckpt', required=True, help='Path to network-snapshot.pth from denoise training')
    ap.add_argument('--guide-ckpt', default='global_diffusion_map/ckpts/guide/network-snapshot.pth', help='Path to guidance network snapshot')
    ap.add_argument('--agg-pred-dir', required=True, help='Dir of aggregated prediction pkls (per scene)')
    ap.add_argument('--bounds-dir', required=False, default=None, help='Optional dir to take scene bounds from (per-scene <id>.pkl); if absent, fall back to agg-pred-dir file or computed from vectors')
    # Conditioning raster root (rendered GT). Prefer 10/11; keep semantic root for backward-compat only.
    ap.add_argument('--cond-root', required=False, default=None, help='Root of conditioning rasters per scene (e.g., rendered_gt/<split>)')
    ap.add_argument('--cond-filename', required=False, default='10_render_gt.png', help='Condition filename inside each scene dir (e.g., 10_render_gt.png or 11_gt_aug.png)')
    ap.add_argument('--semantic-root', required=False, default=None, help='[Deprecated] Root of 08_agg_semantic.png per scene (fallback if cond-root missing)')
    ap.add_argument('--out-dir', required=True, help='Where to save refined pickles')
    ap.add_argument('--scenes', nargs='+', required=False, default=None, help='Optional scene ids; if omitted, infer from agg dir')
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json', help='Stats JSON for M and budgets')
    ap.add_argument('--steps', type=int, default=18, help='Sampling steps (PolyDiffuse default=18)')
    ap.add_argument('--sigma_min', type=float, default=0.002, help='Lowest noise level (EDM)')
    ap.add_argument('--sigma_max', type=float, default=80.0, help='Highest noise level (EDM)')
    ap.add_argument('--rho', type=float, default=7.0, help='Time step exponent (EDM)')
    ap.add_argument('--S_churn', type=float, default=0.0, help='EDM churn')
    ap.add_argument('--S_min', type=float, default=0.0, help='EDM churn min sigma')
    ap.add_argument('--S_max', type=float, default=float('inf'), help='EDM churn max sigma')
    ap.add_argument('--S_noise', type=float, default=1.0, help='EDM churn noise scale')
    ap.add_argument('--second_order', action='store_true', help='Enable 2nd order correction (PolyDiffuse default)')
    ap.add_argument('--identity-output', action='store_true', help='Skip denoising and directly output the (resampled) input vectors')
    # Length filter (per-class, meters). Default 0 disables filtering.
    ap.add_argument('--min-len-ped', type=float, default=0.0, help='Minimum length (m) to keep ped crossings; 0 disables')
    ap.add_argument('--min-len-div', type=float, default=0.0, help='Minimum length (m) to keep dividers; 0 disables')
    ap.add_argument('--min-len-bnd', type=float, default=0.0, help='Minimum length (m) to keep boundaries; 0 disables')
    # Start mode & blending controls
    # Revert to official-aligned default: start from guide anchors
    ap.add_argument('--start-from', choices=['blend', 'proposal', 'guide', 'noise'], default='guide',
                    help='Start state: blend (proposal+guide), proposal only, guide only, or pure noise')
    ap.add_argument('--c_in_override', type=float, default=None,
                    help='Override initial blending c_in in [0,1] (None = use value from sigma_max)')
    ap.add_argument('--proposal-root', required=False, default=None, help='Optional root of packed proposals (<scene>.npz). If set, use these proposals instead of building from agg_pred.')
    # Stable permutation (proposal ↔ guide) matching controls
    ap.add_argument('--stable-permutation', action='store_true', help='Enable classwise matching + gating (per-class) between proposal slots and anchors')
    ap.add_argument('--cost-w-center', type=float, default=1.0, help='Base weight for centroid distance term')
    ap.add_argument('--cost-w-dir', type=float, default=0.2, help='Base weight for direction similarity term')
    ap.add_argument('--cost-w-chamfer', type=float, default=0.5, help='Base weight for simplified Chamfer term')
    ap.add_argument('--match-thres', type=float, default=0.25, help='Base gate on normalized cost (per-class overrides apply)')
    ap.add_argument('--cand-radius-frac', type=float, default=0.6, help='Candidate radius in normalized coords diag fraction (per-class overrides apply)')
    ap.add_argument('--save-steps-dir', type=str, default=None, help='Optional output dir to save per-step PNGs under <dir>/<scene>/step_XX.png')
    ap.add_argument('--viz-steps-with-points', action='store_true', help='Use MapTracker vis_global (points visible) for step viz')
    ap.add_argument('--overlay-condition', action='store_true', help='If saving steps, overlay vectors on conditioning raster')
    # Conditioning raster alignment to training
    ap.add_argument('--cond-max-side', type=int, default=1024, help='Downscale raster to limit max side (None to disable)')
    ap.add_argument('--cond-fixed-size', type=int, nargs=2, default=[1024, 1024], metavar=('H', 'W'), help='Letterbox raster to fixed canvas (omit to disable)')
    ap.add_argument('--disable-proposal-encoder', action='store_true', help='Do not pass proposal_* tensors to the model (ablation).')
    # No extra postprocess by default (keep M points per instance to match training)
    args = ap.parse_args()

    if not args.scenes:
        args.scenes = [osp.splitext(x)[0] for x in os.listdir(args.agg_pred_dir) if x.endswith('.pkl')]
    with open(args.stats_json, 'r') as f:
        s = json.load(f)
    # PolyDiffuse-aligned fallbacks (if stats JSON lacks fields)
    M = int(s.get('M', 20))
    budgets = {int(k): int(v) for k, v in s.get('class_budget', {0: 8, 1: 17, 2: 25}).items()}
    num_queries = int(s.get('num_queries', sum(budgets.values())))

    net, cfg = build_net(args.config, args.ckpt)
    device = next(net.parameters()).device
    # Optional guide network
    net_guide = None
    if args.guide_ckpt and os.path.exists(args.guide_ckpt):
        try:
            net_guide = build_guide(net.model, args.guide_ckpt, device=str(device), num_verts=M, num_queries=num_queries)
        except Exception as e:
            print(f'[warn] failed to load guide net ({args.guide_ckpt}): {e}')

    for scene in args.scenes:
        pred_pkl = osp.join(args.agg_pred_dir, f'{scene}.pkl')
        pred = _load_pickle(pred_pkl)
        # Bounds selection:
        # 1) If --bounds-dir provided, prefer bounds from there (or compute from that file's vectors)
        # 2) Else, use bounds stored in the current pred file
        # 3) Else, compute from current pred vectors
        bounds = None
        if args.bounds_dir is not None:
            ref_pkl = osp.join(args.bounds_dir, f'{scene}.pkl')
            if osp.exists(ref_pkl):
                ref = _load_pickle(ref_pkl)
                if 'bounds' in ref:
                    bounds = ref['bounds']
                else:
                    pts_all = []
                    for k in (0, 1, 2):
                        for arr in ref.get(k, []):
                            pts_all.append(arr)
                    if len(pts_all) > 0:
                        cat = np.concatenate(pts_all, axis=0)
                        minx, miny = cat.min(0)
                        maxx, maxy = cat.max(0)
                        bounds = [float(minx), float(miny), float(maxx), float(maxy)]
        if bounds is None:
            if 'bounds' in pred:
                bounds = pred['bounds']
            else:
                pts_all = []
                for k in (0, 1, 2):
                    for arr in pred.get(k, []):
                        pts_all.append(arr)
                if len(pts_all) == 0:
                    print(f'[warn] empty vectors: {scene}, skipping')
                    continue
                cat = np.concatenate(pts_all, axis=0)
                minx, miny = cat.min(0)
                maxx, maxy = cat.max(0)
                bounds = [float(minx), float(miny), float(maxx), float(maxy)]

        # Build instance arrays with budgets per original class id.
        # Note: even for start-from 'noise' we still need instance slots
        # (labels, masks, bounds) to drive the sampler; the latent itself
        # will be randomized later. Treat 'noise' like 'blend'/'proposal'
        # when constructing inst/labels.
        inst: List[np.ndarray] = []
        labels: List[int] = []
        masks_per_inst: List[np.ndarray] = []  # valid-vertex mask when using packed proposals
        if args.start_from in ('blend', 'proposal', 'noise'):
            # Prefer packed proposals if provided
            used_packed = False
            if getattr(args, 'proposal_root', None):
                pp = osp.join(args.proposal_root, f'{scene}.npz')
                if osp.exists(pp):
                    try:
                        arr = np.load(pp)
                        p_pts = arr['pts'].astype(np.float32)
                        p_msk = arr['mask'].astype(bool)
                        p_labs = arr['labels'].astype(np.int64)
                        Np = min(num_queries, p_pts.shape[0])
                        for j in range(Np):
                            if p_msk[j].all():
                                continue
                            inst.append(p_pts[j])
                            labels.append(int(p_labs[j]))
                            masks_per_inst.append(p_msk[j])
                        used_packed = True
                    except Exception as e:
                        print(f'[warn] failed to use packed proposals for {scene}: {e}')
            if not used_packed:
                # Build from aggregated preds (length‑ranked) with basic gating
                def _curve_len_m(arr: np.ndarray) -> float:
                    a = np.asarray(arr, dtype=np.float32)
                    if a.shape[0] < 2:
                        return 0.0
                    d = a[1:] - a[:-1]
                    return float(np.linalg.norm(d, axis=1).sum())
                # Class mapping: orig->MapTR label: ped(0)->1, divider(1)->0, boundary(2)->2
                # Apply per-class minimum length from args (0 disables)
                min_len = {0: float(args.min_len_div),  # MapTR 0 is divider
                           1: float(args.min_len_ped),  # MapTR 1 is ped
                           2: float(args.min_len_bnd)}  # MapTR 2 is boundary
                for orig in (2, 1, 0):
                    vecs = pred.get(orig, [])
                    lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                    lengths = [(_curve_len_m(v), i) for i, v in enumerate(vecs)]
                    lengths.sort(key=lambda x: x[0], reverse=True)
                    take = int(budgets.get(orig, 0))
                    picked = 0
                    for L, idx in lengths:
                        if L < min_len.get(lab, 0.0):
                            continue
                        smp = _uniform_sample(vecs[idx], M)
                        norm = _normalize_xy(smp, bounds)
                        inst.append(norm)
                        labels.append(lab)
                        picked += 1
                        if picked >= take:
                            break
        elif args.start_from == 'guide':
            # Construct empty instances per class budget; geometry will come from guide
            for orig in (2, 1, 0):
                cap = int(budgets.get(orig, 0))
                for _ in range(cap):
                    # zeros in normalized coords
                    inst.append(np.zeros((M, 2), dtype=np.float32))
                    lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                    labels.append(lab)

        N = min(len(inst), num_queries)
        if N == 0:
            print(f'[warn] no instances after capping: {scene}, skipping')
            continue
        pts = np.zeros((1, num_queries, M, 2), dtype=np.float32)
        mask = np.ones((1, num_queries, M), dtype=bool)
        labs = np.zeros((1, num_queries), dtype=np.int64)
        for i in range(N):
            pts[0, i] = inst[i]
            labs[0, i] = labels[i]
            if i < len(masks_per_inst):
                mask[0, i] = masks_per_inst[i]
            else:
                mask[0, i] = False

        # Conditioning raster (prefer 10/11 under cond-root)
        cond_path = None
        if args.cond_root is not None:
            cp = osp.join(args.cond_root, scene, args.cond_filename)
            if osp.exists(cp):
                cond_path = cp
            else:
                print(f'[warn] missing condition {args.cond_filename} under {scene}; falling back to semantic-root if provided')
        if cond_path is None and args.semantic_root is not None:
            sp = osp.join(args.semantic_root, scene, '08_agg_semantic.png')
            if osp.exists(sp):
                cond_path = sp
                if args.cond_root is None:
                    print(f'[warn] using deprecated 08 as condition for {scene}; set --cond-root/--cond-filename to use 10/11')
        if cond_path is None:
            print(f'[warn] no condition raster for {scene}, skipping')
            continue
        # Load and align raster to match training behavior
        cond_fixed = tuple(args.cond_fixed_size) if args.cond_fixed_size else None
        img = _load_raster(cond_path, cond_max_side=args.cond_max_side, cond_fixed_size=cond_fixed)
        # Build kwargs for net
        metas = [[{
            'img_shape': [(img.shape[-2], img.shape[-1])],
            'lidar2img': np.eye(4, dtype=np.float32)[None, ...],
            'can_bus': np.zeros(18, dtype=np.float32),
        }]]
        model_kwargs = {
            'img': torch.from_numpy(img).unsqueeze(0).to(device),  # add batch
            'poly_class': torch.from_numpy(labs).to(device),
            'poly_mask': torch.from_numpy(mask).to(device),
            'img_metas': metas,
            'cache_image_feat': False,
            'use_cached_feat': False,
        }
        # Provide the same proposals as vector condition (unless ablated or head unsupported)
        head_supports_prop = hasattr(getattr(net, 'model', None), 'pts_bbox_head') and \
                              hasattr(net.model.pts_bbox_head, 'use_proposal') and \
                              bool(getattr(net.model.pts_bbox_head, 'use_proposal'))
        if head_supports_prop and (not args.disable_proposal_encoder) and (args.start_from in ('blend', 'proposal')):
            model_kwargs['proposal_pts'] = torch.from_numpy(pts).to(device)
            model_kwargs['proposal_mask'] = torch.from_numpy(mask.copy()).to(device)
            model_kwargs['proposal_labels'] = torch.from_numpy(labs.copy()).to(device)
        init_pts = torch.from_numpy(pts).to(device)
        # Build mu_guide using PolyMeta guide if available (official alignment)
        if net_guide is not None:
            valid_mask = model_kwargs['poly_mask'][:, :N]
            valid_class = model_kwargs['poly_class'][:, :N]
            valid_pts = init_pts[:, :N]
            guide_attn = torch.zeros_like(valid_mask)
            with torch.no_grad():
                guide_mean_center, guide_sigma = net_guide(valid_pts, guide_attn, valid_class)
                guide_mean = guide_mean_center[:, :, None, :].repeat(1, 1, M, 1)

            # Optional stable permutation: reorder proposal slots to match target slot order (per class)
            if args.stable_permutation and not getattr(args, 'proposal_root', None):
                try:
                    from global_diffusion_map.lib.stable_match import stable_match_reorder as _smr
                    # Anchors: use guide centers (normalized) tiled to M
                    gm = guide_mean_center.detach().cpu().numpy()[0, :N]  # [N,2]
                    tgt = np.tile(gm[:, None, :], (1, M, 1))  # [N,M,2]
                    prop_np = init_pts.detach().cpu().numpy()  # [1,num_queries,M,2]
                    prop_labels = labs[0, :N].astype(np.int64)
                    # Per-class overrides (MapTR labels: 0=divider,1=ped,2=boundary)
                    per_class = {
                        0: {'w_center': max(1.0, float(args.cost_w_center)), 'w_dir': 0.15, 'w_pw': 0.30,
                            'thres': max(0.22, float(args.match_thres)-0.10), 'cand_radius_frac': max(0.40, float(args.cand_radius_frac)-0.20), 'pw_thres': 0.10},
                        1: {'w_center': max(1.0, float(args.cost_w_center)), 'w_dir': 0.15, 'w_pw': 0.30,
                            'thres': max(0.22, float(args.match_thres)-0.08), 'cand_radius_frac': max(0.45, float(args.cand_radius_frac)-0.15), 'pw_thres': 0.12},
                        2: {'w_center': 2.0, 'w_dir': 0.05, 'w_pw': 0.20,
                            'thres': max(0.22, float(args.match_thres)-0.10), 'cand_radius_frac': max(0.35, float(args.cand_radius_frac)-0.25), 'pw_thres': 0.15},
                    }
                    new_pts, new_mask = _smr(
                        prop_np=prop_np[:, :N],
                        tgt_np=tgt[None, ...],
                        labs_np=labs[:, :N],
                        mask_np=mask[:, :N],
                        prop_labels=prop_labels,
                        w_center=float(args.cost_w_center),
                        w_dir=float(args.cost_w_dir),
                        w_pw=float(args.cost_w_chamfer),
                        thres=float(args.match_thres),
                        cand_radius_frac=float(args.cand_radius_frac),
                        m_pw=8,
                        per_class=per_class,
                    )
                    init_pts[:, :N] = torch.from_numpy(new_pts).to(device)
                    model_kwargs['poly_mask'][:, :N] = torch.from_numpy(new_mask).to(device)
                    # Keep ProposalEncoder aligned to the same slot order
                    if 'proposal_pts' in model_kwargs and isinstance(model_kwargs['proposal_pts'], torch.Tensor):
                        # model_kwargs['proposal_pts'] has shape [1,num_queries,M,2]
                        pp = model_kwargs['proposal_pts']
                        pp[:, :N] = torch.from_numpy(new_pts).to(pp.device)
                        model_kwargs['proposal_pts'] = pp
                    if 'proposal_mask' in model_kwargs and isinstance(model_kwargs['proposal_mask'], torch.Tensor):
                        pm = model_kwargs['proposal_mask']
                        pm[:, :N] = torch.from_numpy(new_mask).to(pm.device)
                        model_kwargs['proposal_mask'] = pm
                    # Post-gate by world centroid distance vs guide center
                    # Denormalize guide centers and curves
                    g_centers = guide_mean_center.detach().cpu().numpy()[0, :N]  # [N,2] normalized
                    g_world = _denormalize_xy(g_centers, bounds)
                    np_pts = init_pts.detach().cpu().numpy()
                    np_mask = model_kwargs['poly_mask'].detach().cpu().numpy()
                    cmax = {0: 4.0, 1: 4.0, 2: 5.0}
                    for j in range(N):
                        if np_mask[0, j].all():
                            continue
                        labj = int(labs[0, j])
                        crv = _denormalize_xy(np_pts[0, j], bounds)
                        p = crv.mean(axis=0)
                        d = float(np.linalg.norm(p - g_world[j]))
                        if d > cmax.get(labj, 5.0):
                            np_mask[0, j] = True
                    model_kwargs['poly_mask'] = torch.from_numpy(np_mask).to(device)
                    # Keep proposal_mask consistent with final gating
                    if 'proposal_mask' in model_kwargs and isinstance(model_kwargs['proposal_mask'], torch.Tensor):
                        model_kwargs['proposal_mask'] = torch.from_numpy(np_mask).to(device)
                except Exception as e:
                    print(f"[warn] stable permutation failed, continue without: {e}")

            # Determine initial mixing c_in for start state if using 'blend'
            if args.c_in_override is not None:
                c_in0 = torch.as_tensor(float(args.c_in_override), device=device)
            else:
                sigma_max = torch.as_tensor(float(args.sigma_max), device=device)
                c_in0 = 1.0 / torch.sqrt(net.sigma_data ** 2 + sigma_max ** 2)

            if args.start_from == 'noise':
                # Start from Gaussian noise; keep mu_guide as guidance only
                latents = torch.randn_like(init_pts)
                mu_guide = init_pts.clone(); mu_guide[:, :N] = guide_mean
            elif args.start_from == 'guide':
                mu_guide = init_pts.clone(); mu_guide[:, :N] = guide_mean
                latents = mu_guide.clone()
            elif args.start_from == 'proposal':
                mu_guide = init_pts.clone()
                latents = init_pts.clone()
            else:
                mixed = c_in0 * init_pts[:, :N] + (1.0 - c_in0) * guide_mean
                mu_guide = init_pts.clone(); mu_guide[:, :N] = mixed
                latents = mu_guide.clone()
        else:
            # Without guide: keep current behavior
            mu_guide = init_pts.clone()
            latents = init_pts.clone()

        # No pixel gating in baseline; keep masks from matching/gating above

        if args.identity_output:
            pred_np = init_pts.detach().cpu().numpy()
            pred_pts = None
        else:
            pred_pts = edm_sampler(
                net,
                latents,
                mu_guide,
                model_kwargs,
                num_steps=args.steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                S_churn=args.S_churn,
                S_min=args.S_min,
                S_max=args.S_max,
                S_noise=args.S_noise,
                second_order=args.second_order or True,
                keep_intermediates=bool(args.save_steps_dir),
            )
            pred_np = pred_pts[-1].detach().cpu().numpy() if isinstance(pred_pts, list) else pred_pts.detach().cpu().numpy()
        # pred_np shape [B, N, M, 2]

        # Denormalize and drop masked/short curves, then keep top-K per class by length (budget caps)
        out = {0: [], 1: [], 2: [], 'bounds': bounds}
        back_map = {0: 1, 1: 0, 2: 2}  # MapTR -> original keys
        # Use latest mask after any gating/permutation
        try:
            final_mask_np = model_kwargs['poly_mask'].detach().cpu().numpy()[0]
        except Exception:
            final_mask_np = mask[0]
        # Per-class minimal length (meters) to keep; defaults are 0 (disabled)
        min_len_cls = {0: float(args.min_len_ped), 1: float(args.min_len_div), 2: float(args.min_len_bnd)}
        Q = pred_np.shape[1]
        lengths_world = {0: [], 1: [], 2: []}
        for i in range(Q):
            # skip if fully masked
            if final_mask_np[i].all():
                continue
            lab_i = int(labs[0, i])
            # only use valid vertices
            valid = ~final_mask_np[i]
            curve_norm = pred_np[0, i][valid]
            if curve_norm.shape[0] < 2:
                continue
            curve_world = _denormalize_xy(curve_norm, bounds)
            # length filter
            d = curve_world[1:] - curve_world[:-1]
            L = float(np.linalg.norm(d, axis=1).sum())
            if L < min_len_cls.get(lab_i, 0.0):
                continue
            mapped_label = back_map.get(lab_i, 2)
            out[mapped_label].append(curve_world.astype(np.float32))
            lengths_world[mapped_label].append(L)

        # Per-class budget cap: keep top-K by length
        try:
            # budgets dict from stats JSON uses original keys (0:ped,1:divider,2:boundary)
            cap0 = int(budgets.get(0, len(out[0])))
            cap1 = int(budgets.get(1, len(out[1])))
            cap2 = int(budgets.get(2, len(out[2])))
            for cls_id, cap in [(0, cap0), (1, cap1), (2, cap2)]:
                if len(out[cls_id]) > cap:
                    order = sorted(range(len(out[cls_id])), key=lambda k: lengths_world[cls_id][k], reverse=True)
                    keep_idx = set(order[:cap])
                    out[cls_id] = [out[cls_id][k] for k in range(len(out[cls_id])) if k in keep_idx]
        except Exception:
            pass

        out_pkl = osp.join(args.out_dir, f'{scene}.pkl')
        _save_pickle(out, out_pkl)
        print(f'[ok] {scene} -> {out_pkl}')

        # Optional per-step visualization
        if (pred_pts is not None) and isinstance(pred_pts, list) and args.save_steps_dir:
            steps_dir = osp.join(args.save_steps_dir, scene)
            os.makedirs(steps_dir, exist_ok=True)
            # Map back from MapTR label (0:divider,1:ped,2:boundary) to original keys (0:ped,1:divider,2:boundary)
            back_map = {0: 1, 1: 0, 2: 2}
            valid_N = N
            if args.viz_steps_with_points:
                # Use MapTracker vis_global to plot with points visible
                import importlib.util as _ilu
                import sys as _sys
                _vis_dir = osp.join(osp.dirname(__file__), '..', '..', 'maptracker', 'tools', 'visualization')
                _vis_dir = osp.abspath(_vis_dir)
                if _vis_dir not in _sys.path:
                    _sys.path.insert(0, _vis_dir)
                from vis_global import plot_fig_unmerged  # type: ignore
                class _Args:
                    def __init__(self, dpi:int):
                        self.transparent=False; self.dpi=dpi
                _viz_args=_Args(60)
                car_traj=[[np.array([0.0,0.0]),0.0]]
                minx, miny, maxx, maxy = bounds
                x_min, x_max, y_min, y_max = float(minx), float(maxx), float(miny), float(maxy)
                for si, tensor in enumerate(pred_pts):
                    arr = tensor.detach().cpu().numpy()  # [B,N,M,2]
                    curves = arr[0, :valid_N]
                    # Build bank per step
                    bank: Dict[str, List[np.ndarray]] = {}
                    for q in range(valid_N):
                        cls_maptr = int(labs[0, q])
                        orig_cls = back_map.get(cls_maptr, 2)
                        xy = _denormalize_xy(curves[q], bounds)
                        bank[f"{orig_cls}_{q}"] = [xy.astype(np.float32)]
                    out_png = osp.join(steps_dir, f'step_{si:02d}.png')
                    try:
                        plot_fig_unmerged(car_traj, x_min, x_max, y_min, y_max, out_png, bank, _viz_args)
                    except Exception as e:
                        print(f"[warn] vis_global step plot failed: {e}")
            else:
                # Lightweight Matplotlib renderer
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                color = {0: 'b', 1: 'r', 2: 'g'}  # ped, divider, boundary
                minx, miny, maxx, maxy = bounds
                for si, tensor in enumerate(pred_pts):
                    arr = tensor.detach().cpu().numpy()
                    curves = arr[0, :valid_N]
                    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(1, 1, 1)
                    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy); ax.set_aspect('equal', adjustable='box'); ax.set_facecolor('white'); ax.axis('off')
                    for q in range(valid_N):
                        cls_maptr = int(labs[0, q]); orig_cls = back_map.get(cls_maptr, 2)
                        xy = _denormalize_xy(curves[q], bounds)
                        ax.plot(xy[:,0], xy[:,1], color=color.get(orig_cls,'k'), linewidth=1.5, alpha=0.9)
                    out_png = osp.join(steps_dir, f'step_{si:02d}.png')
                    plt.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


if __name__ == '__main__':
    main()
