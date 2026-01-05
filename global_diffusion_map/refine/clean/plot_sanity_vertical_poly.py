#!/usr/bin/env python
from __future__ import annotations

"""
Plot a 4-panel vertical sanity figure for a single scene:
  (1) Ground-truth global vectors (meters, static GT)
  (2) Corrupted input (aggregated proposal, meters)
  (3) Refinement from proposal with raster condition
  (4) Refinement from proposal without raster (blind test)

要求：
- 不叠加 letterboxed 条件图，只画 polyline；
- 颜色与 viz_global 对齐：divider=红, ped=蓝, boundary=绿, overlap=青（这里只用前三类）。

依赖：
- 先用 train_one_scene_polydiffuse_encoder_stable_matching.py 训练得到 ckpt；
- 再用 infer_polydiffuse_aligned.py 分别跑有条件/盲测推理，并写出
  {scene}_refined_m.pkl / {scene}_input_m.pkl。
"""

import argparse
import os
import os.path as osp
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys  # noqa: E402
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.single_scene_overfit import load_pickle  # noqa: E402


def _load_vec_pkl(path: str) -> Dict[int, List[np.ndarray]]:
    d = load_pickle(path)
    out: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    for cid in (0, 1, 2):
        arrs = d.get(cid, []) or []
        out[cid] = [np.asarray(a, dtype=np.float32) for a in arrs]
    return out


def _plot_panel(ax: plt.Axes,
                bank: Dict[int, List[np.ndarray]],
                bounds: Sequence[float],
                title: str) -> None:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    # 颜色：RGB，与 viz_global 语义一致
    # ped(0)=蓝, divider(1)=红, boundary(2)=绿
    color_map = {0: (0.0, 0.0, 1.0), 1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0)}
    for cid in (0, 1, 2):
        col = color_map[cid]
        for arr in bank.get(cid, []):
            if arr is None or len(arr) < 2:
                continue
            xy = np.asarray(arr, dtype=np.float32)
            ax.plot(xy[:, 0], xy[:, 1], color=col, linewidth=1.5)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def main() -> None:
    ap = argparse.ArgumentParser(description="One-scene PolyDiffuse sanity vertical figure (polylines only)")
    ap.add_argument("--static-root", required=True, help="static GT root: .../static_gt_vector/av2_oldsplit/val")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--infer-cond-root", required=True,
                    help="输出根目录（有条件推理），例如 infer_polydiff/.../proposal")
    ap.add_argument("--infer-blind-root", required=True,
                    help="输出根目录（盲测推理），例如 infer_polydiff/.../proposal_blind")
    ap.add_argument("--out-root", default="global_diffusion_map/refine/work_dirs/sanity_figs")
    args = ap.parse_args()

    scene = args.scene

    # 1) 加载静态 GT （meters）
    gt_pkl = osp.join(args.static_root, f"{scene}.pkl")
    gt = load_pickle(gt_pkl)
    if "bounds" not in gt or gt["bounds"] is None:
        raise RuntimeError(f"bounds-missing in static GT: {gt_pkl}")
    bounds = [float(v) for v in gt["bounds"]]
    gt_bank: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    for cid in (0, 1, 2):
        arrs = gt.get(cid, []) or []
        gt_bank[cid] = [np.asarray(a, dtype=np.float32) for a in arrs]

    # 2) 加载 corrupted input（proposal）和 refined（cond / blind）
    # infer_polydiffuse_aligned.py 会在各自 out_dir 下写出 {scene}_input_m.pkl / {scene}_refined_m.pkl
    inp_cond_pkl = osp.join(args.infer_cond_root, f"{scene}_input_m.pkl")
    ref_cond_pkl = osp.join(args.infer_cond_root, f"{scene}_refined_m.pkl")
    inp_blind_pkl = osp.join(args.infer_blind_root, f"{scene}_input_m.pkl")
    ref_blind_pkl = osp.join(args.infer_blind_root, f"{scene}_refined_m.pkl")

    # 输入 proposal 在 cond/blind 两次推理中理论相同，只要读取一份即可；为了稳妥，若 blind 版本存在也可任选其一。
    if not osp.exists(inp_cond_pkl) and osp.exists(inp_blind_pkl):
        inp_pkl = inp_blind_pkl
    else:
        inp_pkl = inp_cond_pkl

    if not osp.exists(inp_pkl):
        raise FileNotFoundError(f"input_m.pkl not found under {args.infer_cond_root} or {args.infer_blind_root}")
    if not osp.exists(ref_cond_pkl):
        raise FileNotFoundError(f"refined_m.pkl (cond) not found: {ref_cond_pkl}")
    if not osp.exists(ref_blind_pkl):
        raise FileNotFoundError(f"refined_m.pkl (blind) not found: {ref_blind_pkl}")

    inp_bank = _load_vec_pkl(inp_pkl)
    ref_cond_bank = _load_vec_pkl(ref_cond_pkl)
    ref_blind_bank = _load_vec_pkl(ref_blind_pkl)

    # 3) 画 4 个子图（2x2，从左上到右下依次为 1–4）
    fig_h = 8
    fig_w = 8
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), constrained_layout=True)
    titles = [
        "Ground Truth",
        "Proposal",
        "Refinement with Condition",
        "Refinement without Condition (Blind)",
    ]
    banks = [gt_bank, inp_bank, ref_cond_bank, ref_blind_bank]
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax, bank, title in zip(axes_flat, banks, titles):
        _plot_panel(ax, bank, bounds, title)

    out_dir = osp.join(args.out_root, scene)
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, "sanity_check_vertical.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[ok] sanity figure saved: {out_path}")


if __name__ == "__main__":
    main()
