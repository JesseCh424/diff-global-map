#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os.path as osp
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.single_scene_dataset import SingleSceneDataset
from global_diffusion_map.refine.dataset_refine import RefineCaps
from global_diffusion_map.refine.loss_refine import criterion
from global_diffusion_map.refine.single_scene_overfit import RasterEncoder, overlay_on_raster, compute_bounds_from_any, load_pickle
from global_diffusion_map.refine.model_refine import SlotMLPWithTime


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main() -> None:
    ap = argparse.ArgumentParser(description='Refine training (SDEdit-like) on single scene')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--agg-pred-root', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--epochs', type=int, default=800)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=1.5)
    ap.add_argument('--alpha-const', type=float, default=0.03, help='若提供，则使用固定 alpha 做部分加噪（覆盖 sigma 采样映射）')
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_refine')
    ap.add_argument('--resume-ckpt', type=str, default=None, help='可选：从 ckpt_ep_xxxx.pth 继续训练（仅加载权重）')
    args = ap.parse_args()

    set_seed(0)
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    caps = RefineCaps(num_queries=N, num_points=P)

    ds = SingleSceneDataset(
        static_root=args.static_root,
        rendered_root=args.rendered_root,
        agg_pred_root=args.agg_pred_root,
        scene=args.scene,
        caps=caps,
        class_budgets=budgets,
        length=max(2000, args.epochs * 2),
        jitter_sigma_m=0.6,
        drop_rate=0.2,
        ghosts=1,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = RasterEncoder(out_dim=256).to(device)
    net = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64).to(device)
    opt = optim.AdamW(list(enc.parameters()) + list(net.parameters()), lr=args.lr, weight_decay=args.wd)

    out_dir = osp.join(args.out_root, args.scene)
    viz_dir = osp.join(out_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    # 选择性恢复权重（仅权重），优化器按新的 lr 初始化
    if args.resume_ckpt and osp.isfile(args.resume_ckpt):
        try:
            data = torch.load(args.resume_ckpt, map_location=device)
            if 'encoder' in data:
                enc.load_state_dict(data['encoder'], strict=False)
            if 'net' in data:
                net.load_state_dict(data['net'], strict=False)
            print(f"[resume] loaded weights from {args.resume_ckpt}")
        except Exception as e:
            print(f"[warn] failed to resume weights: {e}")

    enc.train(); net.train()
    for ep in range(1, args.epochs + 1):
        sample = ds[ep - 1]
        # Proposal (simulated) + partial noise (SDEdit-like)
        x = sample['proposal'].numpy()  # [N,P,2]
        if args.alpha_const is not None:
            alpha = float(args.alpha_const)
        else:
            sigma = float(np.random.uniform(args.sigma_min, args.sigma_max))
            alpha = min(1.0, max(0.0, (sigma - args.sigma_min) / max(args.sigma_max - args.sigma_min, 1e-6)))
        x_t = np.clip((1.0 - alpha) * x + alpha * np.random.normal(size=x.shape).astype(np.float32), -1.0, 1.0)
        x_t = torch.from_numpy(x_t[None, ...]).to(device)
        r = sample['raster'][None, ...].to(device)
        tgt_c = sample['tgt_coords'][None, ...].to(device)
        tgt_m = sample['tgt_mask'][None, ...].to(device)
        tgt_p = sample['tgt_present'][None, ...].to(device)

        rv = enc(r)
        # 用 alpha 替代 sigma 作为“噪声强度标量”输入（可选）；为了兼容，仍保留张量名 t_scalar
        t_scalar = torch.full((x_t.shape[0],), fill_value=float(alpha), device=device, dtype=torch.float32)
        pred_coords, pred_logits = net(x_t, rv, t_scalar)

        losses = criterion(pred_coords, pred_logits, tgt_c, tgt_m, tgt_p, l1_weight=1.0, cls_weight=1.0)
        loss = losses['loss_cls'] + losses['loss_reg']
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % 20 == 0 or ep == 1:
            print(f"[ep {ep:04d}] alpha={alpha:.4f} loss={loss.item():.6f} cls={losses['loss_cls'].item():.6f} reg={losses['loss_reg'].item():.6f}")
        if ep % 100 == 0 or ep == args.epochs:
            with torch.no_grad():
                pc = pred_coords.detach().cpu().numpy()[0]
                pl = pred_logits.detach().cpu().numpy()[0].reshape(-1)
                prob = 1.0 / (1.0 + np.exp(-pl))
                keep = np.where(prob >= 0.5)[0].tolist()
                pred_list = [pc[i] for i in keep]
                # 可视化叠加（对齐 10）
                # 需要 bounds 与栅格：从数据集中恢复
                gt = load_pickle(osp.join(args.static_root, f'{args.scene}.pkl'))
                bounds = gt.get('bounds')
                if bounds is None:
                    agg = load_pickle(osp.join(args.agg_pred_root, f'{args.scene}.pkl'))
                    b2 = agg.get('bounds', None)
                    if b2 is None:
                        raise RuntimeError(
                            'bounds-missing: static GT has no bounds and aggregated pred also missing bounds.\n'
                            f'  static_pkl={osp.join(args.static_root, f"{args.scene}.pkl")}\n'
                            f'  agg_pred_pkl={osp.join(args.agg_pred_root, f"{args.scene}.pkl")}')
                    bounds = b2
                overlay_on_raster(osp.join(viz_dir, f'ep_{ep:04d}.png'), ds.raster, bounds, ds.gt_pack, pred_list)
                # Save checkpoint
                ckpt = {
                    'encoder': enc.state_dict(),
                    'net': net.state_dict(),
                    'P': P, 'N': N,
                    'budgets': budgets,
                }
                os.makedirs(out_dir, exist_ok=True)
                torch.save(ckpt, osp.join(out_dir, f'ckpt_ep_{ep:04d}.pth'))

    print(f"[ok] refine training finished. Visualizations under {viz_dir}. Checkpoints under {out_dir}")


if __name__ == '__main__':
    main()
