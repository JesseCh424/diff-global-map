#!/usr/bin/env python
from __future__ import annotations

"""
Phase‑B Generation Evaluation (multi‑scene).

Checks whether the Phase‑B model can denoise inputs toward the condition map
on (unseen) scenes, for three starts: proposal, gt_noise, random.

Outputs per scene:
  - final_pred.pkl (coords/logits/sem/bounds/meta)
  - refined.png (pred over raster), proposal.png (if available), gt.png
  - metrics.json (per scene) and aggregates to metrics.jsonl / metrics.csv

Defaults target the ResNet encoder and 512×512 input to match optimized training.
"""

import argparse
import os
import os.path as osp
import json
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import (
    RefineCaps,
    pack_gt_to_slots,
    pack_vectors_to_slots,
)
from global_diffusion_map.refine.loss_refine import HungarianMatcher
from global_diffusion_map.refine.single_scene_overfit import (
    load_pickle,
    denorm_xy,
    overlay_slots_annot,
)
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule, edm_unrolled_train


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _labels_from_budgets(budgets: Dict[int, int], N: int) -> np.ndarray:
    order: List[int] = []
    for orig in (1, 0, 2):  # MapTR ids: divider=1,ped=0,boundary=2 → labels 0,1,2
        cap = int(budgets.get(orig, 0))
        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
        order += [lab] * max(0, cap)
    out = np.full((N,), -1, dtype=np.int64)
    m = min(N, len(order))
    if m > 0:
        out[:m] = np.asarray(order[:m], dtype=np.int64)
    return out


class StandardResNetEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, version: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        import torchvision.models as models
        # robust weights handling for different torchvision versions
        try:
            if version == 'resnet18':
                m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None); d = 512
            elif version == 'resnet34':
                m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None); d = 512
            elif version == 'resnet50':
                m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None); d = 2048
            else:
                raise ValueError
        except Exception:
            if version == 'resnet18':
                m = models.resnet18(pretrained=pretrained); d = 512
            elif version == 'resnet34':
                m = models.resnet34(pretrained=pretrained); d = 512
            elif version == 'resnet50':
                m = models.resnet50(pretrained=pretrained); d = 2048
            else:
                raise
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.proj = nn.Linear(d, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x).flatten(1))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _simple_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a[np.isfinite(a).all(axis=-1)]; b = b[np.isfinite(b).all(axis=-1)]
    if a.size == 0 or b.size == 0:
        return 0.0
    m = min(a.shape[0], b.shape[0], 16)
    ia = np.linspace(0, a.shape[0] - 1, num=m).round().astype(int)
    ib = np.linspace(0, b.shape[0] - 1, num=m).round().astype(int)
    aa = a[ia]; bb = b[ib]
    da = np.sqrt(((aa[:, None, :] - bb[None, :, :]) ** 2).sum(-1)).min(1).mean()
    db = np.sqrt(((bb[:, None, :] - aa[None, :, :]) ** 2).sum(-1)).min(1).mean()
    return float(0.5 * (da + db))


def _set_chamfer_meter(pred_list: List[np.ndarray], gt_list: List[np.ndarray], bounds: Sequence[float]) -> float:
    # compute symmetric set chamfer in meters (denorm first)
    if len(pred_list) == 0 and len(gt_list) == 0:
        return 0.0
    if len(pred_list) == 0:
        return float(np.mean([0.0 for _ in gt_list]))
    if len(gt_list) == 0:
        return float(np.mean([0.0 for _ in pred_list]))
    # denorm
    Pm = [denorm_xy(p, bounds) for p in pred_list]
    Gm = [denorm_xy(g, bounds) for g in gt_list]
    A = np.mean([min((_simple_chamfer(a, b) for b in Gm), default=0.0) for a in Pm])
    B = np.mean([min((_simple_chamfer(b, a) for a in Pm), default=0.0) for b in Gm])
    return float(0.5 * (A + B))


def _filter_by_thr(coords: np.ndarray, logits: np.ndarray, thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # coords: [N,P,2], logits: [N]
    probs = _sigmoid(logits.reshape(-1))
    keep = probs >= float(thr)
    return coords[keep], probs[keep], keep


def _read_scenes(scenes: List[str] | None, scenes_file: Optional[str]) -> List[str]:
    if scenes and len(scenes) > 0:
        return list(scenes)
    if scenes_file and osp.exists(scenes_file):
        with open(scenes_file, 'r') as f:
            return [ln.strip() for ln in f if ln.strip()]
    raise RuntimeError('no scenes provided')


def main() -> None:
    ap = argparse.ArgumentParser(description='Phase‑B generation evaluation (multi‑scene)')
    # Data
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--agg-pred-root', default='', help='required for start=proposal')
    ap.add_argument('--stats-json', default='global_diffusion_map/refine/work_dirs/av2_stats.json')
    ap.add_argument('--scenes', nargs='+', default=None)
    ap.add_argument('--scenes-file', default='')
    # Encoder / input
    ap.add_argument('--encoder', choices=['resnet', 'raster', 'maptr'], default='resnet')
    ap.add_argument('--resnet', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet50')
    ap.add_argument('--img-size', type=int, default=512)
    # Sampler
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--start', choices=['proposal', 'gt_noise', 'random'], default='proposal')
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.4)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--second-order', action='store_true')
    ap.add_argument('--start-sigma', type=float, default=0.0, help='SDEdit start noise (norm units)')
    # Postprocess
    ap.add_argument('--thr', type=float, default=0.2)
    ap.add_argument('--nms-meters', type=float, default=0.0)
    ap.add_argument('--samples', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    # Output
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/infer_phaseB_eval')

    args = ap.parse_args()
    set_seed(int(args.seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Caps / budgets
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20)); N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    labels_default = _labels_from_budgets(budgets, N)
    RefineCaps(num_points=P, num_queries=N)

    # Build encoder + net and load ckpt
    if args.encoder == 'resnet':
        enc = StandardResNetEncoder(out_dim=256, version=args.resnet, pretrained=False).to(device)
    elif args.encoder == 'raster':
        from global_diffusion_map.refine.single_scene_overfit import RasterEncoder
        enc = RasterEncoder(out_dim=256).to(device)
    else:
        from global_diffusion_map.refine.clean.infer_polydiffuse_aligned import PolyDiffuseImageEncoder256
        enc = PolyDiffuseImageEncoder256('official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py',
                                         'global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth', device=str(device))
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if 'encoder' in ckpt:
        try:
            enc.load_state_dict(ckpt['encoder'], strict=True)
        except Exception:
            enc.load_state_dict(ckpt['encoder'], strict=False)
    if 'net' in ckpt:
        ok = False
        try:
            net.load_state_dict(ckpt['net'], strict=True)
            ok = True
        except Exception:
            pass
        if not ok:
            try:
                net.backbone.load_state_dict(ckpt['net'], strict=False)
            except Exception:
                pass
    enc.eval(); net.eval()

    scenes = _read_scenes(args.scenes, args.scenes_file)
    os.makedirs(args.out_root, exist_ok=True)
    metrics_out = osp.join(args.out_root, 'metrics.jsonl')
    with open(metrics_out, 'w') as _:
        pass

    for scene in scenes:
        try:
            gt = load_pickle(osp.join(args.static_root, f'{scene}.pkl'))
            bounds = gt.get('bounds')
            if bounds is None:
                raise RuntimeError('bounds missing in static pkl')
            gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, budgets, num_points=P, num_queries=N)

            # Raster
            raw = Image.open(osp.join(args.rendered_root, scene, '10_render_gt.png')).convert('RGB')
            if int(args.img_size) > 0:
                raw = raw.resize((int(args.img_size), int(args.img_size)), Image.BILINEAR)
            img_t = TF.normalize(TF.to_tensor(raw), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)

            # Start x0
            if args.start == 'proposal':
                if not args.agg_pred_root:
                    raise RuntimeError('proposal mode requires --agg-pred-root')
                agg = load_pickle(osp.join(args.agg_pred_root, f'{scene}.pkl'))
                x_prop, m_prop, labs = pack_vectors_to_slots(agg, bounds, budgets, num_points=P, num_queries=N)
                input_labels = labs
                x0 = x_prop
            elif args.start == 'gt_noise':
                x0 = gt_pack.copy(); m_prop = gt_mask.copy(); input_labels = labels_default.copy()
            else:
                x0 = np.random.uniform(-1.0, 1.0, size=(N, P, 2)).astype(np.float32)
                m_prop = np.zeros((N, P), dtype=bool); input_labels = labels_default.copy(); input_labels[:] = -1

            # torch tensors
            x0_t = torch.from_numpy(x0).float().unsqueeze(0).to(device)
            ras_t = img_t.unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                rv = enc(ras_t)
            sigmas = karras_schedule(int(args.steps), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)
            if float(args.start_sigma) > 0.0:
                xK = torch.clamp(x0_t + torch.randn_like(x0_t) * float(args.start_sigma), -1.0, 1.0)
            else:
                xK = x0_t.clone()
            x_prior = torch.from_numpy(x0).float().unsqueeze(0).to(device)
            input_labels_t = torch.from_numpy(input_labels).long().unsqueeze(0).to(device)

            # Denoise
            with torch.no_grad(), torch.cuda.amp.autocast():
                pred_coords, pred_logits, preds, _ = edm_unrolled_train(
                    net, xK, rv, sigmas, second_order=bool(args.second_order),
                    cond_prior=x0_t, input_labels=input_labels_t,
                    collect_states=False, collect_preds=True,
                )

            pc = pred_coords[0].detach().cpu().numpy()  # [N,P,2]
            pl = pred_logits[0].detach().cpu().numpy()   # [N]
            # Optional semantic logits at final step
            sem_np: Optional[np.ndarray] = None
            try:
                if preds and len(preds[-1]) > 2 and preds[-1][2] is not None:
                    sem_np = preds[-1][2][0].detach().cpu().numpy()  # [N,C]
            except Exception:
                sem_np = None
            keep_coords, keep_probs, keep_mask = _filter_by_thr(pc, pl, float(args.thr))

            # Metrics (overall chamfer and, if proposal, improvement vs input)
            gt_list = [gt_pack[i] for i in range(N) if not gt_mask[i].all()]
            pred_list = [pc[i] for i in range(N) if keep_mask[i]]
            ch_all = _set_chamfer_meter(pred_list, gt_list, bounds)
            metrics: Dict[str, float] = {'Chamfer/all': ch_all}
            if args.start in ('proposal', 'gt_noise'):
                inp_list = [x0[i] for i in range(N) if not m_prop[i].all()] if args.start == 'proposal' else [gt_pack[i] for i in range(N) if not gt_mask[i].all()]
                ch_inp = _set_chamfer_meter(inp_list, gt_list, bounds)
                metrics['Chamfer/input'] = ch_inp

            # Save overlays
            raster_np = np.asarray(raw, dtype=np.float32).transpose(2, 0, 1) / 255.0
            out_dir = osp.join(args.out_root, scene)
            os.makedirs(out_dir, exist_ok=True)
            # Visualize only confident slots and color by predicted class when available
            draw_mask = np.ones((N, P), dtype=bool)
            labels_draw = None
            if keep_mask.shape[0] == N:
                draw_mask[keep_mask] = False
                if sem_np is not None and sem_np.shape[0] == N:
                    labels_draw = np.argmax(sem_np, axis=-1).astype(np.int64)
            overlay_slots_annot(osp.join(out_dir, 'refined.png'), raster_np, bounds,
                                 slots=pc, mask=draw_mask, labels=labels_draw,
                                 gt_slots=gt_pack, gt_mask=gt_mask)
            if args.start == 'proposal':
                overlay_slots_annot(osp.join(out_dir, 'proposal.png'), raster_np, bounds,
                                     slots=x0, mask=m_prop, labels=input_labels, gt_slots=gt_pack, gt_mask=gt_mask)
            else:
                overlay_slots_annot(osp.join(out_dir, f'{args.start}.png'), raster_np, bounds,
                                     slots=x0, mask=(gt_mask if args.start=='gt_noise' else np.zeros((N,P),bool)),
                                     labels=None, gt_slots=gt_pack, gt_mask=gt_mask)
            overlay_slots_annot(osp.join(out_dir, 'gt.png'), raster_np, bounds,
                                 slots=gt_pack, mask=gt_mask, labels=None)

            # Save pkl + metrics
            import pickle
            with open(osp.join(out_dir, 'final_pred.pkl'), 'wb') as f:
                pickle.dump({
                    'coords': pc, 'logits': pl, 'bounds': bounds, 'thr': float(args.thr),
                    'probs': _sigmoid(pl), 'keep_mask': keep_mask.astype(np.bool_),
                }, f)
            with open(osp.join(out_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            with open(metrics_out, 'a') as f:
                rec = {'scene': scene}; rec.update(metrics); f.write(json.dumps(rec) + '\n')
        except Exception as e:
            print(f"[warn] scene {scene}: {e}")


if __name__ == '__main__':
    main()
