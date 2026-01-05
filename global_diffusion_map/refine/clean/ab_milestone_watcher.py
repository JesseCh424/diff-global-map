#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import os.path as osp
import re
import subprocess
import sys
import time
from typing import Dict, List, Tuple


def parse_metrics_from_stream(stream_lines: List[str]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Parse metrics from combined stdout of compare script calls.
    Returns mapping (mode, encoder)->metrics dict.
    We detect encoder context from compare runner's [run] lines.
    """
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    cur_mode = None
    cur_enc = None  # 'clean' or 'poly'
    # detect mode by compare runner echo: "=== Mode: noise ==="
    mode_pat = re.compile(r"^=== Mode: (\w+) ===")
    run_clean_pat = re.compile(r"\binfer_one_scene_clean.py\b")
    run_poly_pat = re.compile(r"\binfer_polydiffuse_aligned.py\b")
    kv_pat = re.compile(r"^(Ref_vs_[A-Za-z_]+):\s*([0-9.]+|nan)")
    for ln in stream_lines:
        ln = ln.strip()
        m = mode_pat.search(ln)
        if m:
            cur_mode = m.group(1)
            cur_enc = None
            continue
        if 'python -u' in ln and run_clean_pat.search(ln):
            cur_enc = 'clean'
            continue
        if 'python -u' in ln and run_poly_pat.search(ln):
            cur_enc = 'poly'
            continue
        mkv = kv_pat.match(ln)
        if mkv and cur_mode and cur_enc:
            k = mkv.group(1)
            try:
                v = float(mkv.group(2))
            except Exception:
                v = float('nan')
            out.setdefault((cur_mode, cur_enc), {})[k] = v
    return out


def append_metrics_csv(csv_path: str, epoch: int, metrics: Dict[Tuple[str, str], Dict[str, float]]) -> None:
    header = [
        'epoch','mode','encoder',
        'Ref_vs_GT_all_Chamfer_m','Ref_vs_GT_ped_Chamfer_m','Ref_vs_GT_div_Chamfer_m','Ref_vs_GT_bnd_Chamfer_m',
        'Ref_vs_Input_all_Chamfer_m','Ref_vs_Input_ped_Chamfer_m','Ref_vs_Input_div_Chamfer_m','Ref_vs_Input_bnd_Chamfer_m',
    ]
    exists = osp.exists(csv_path)
    with open(csv_path, 'a') as f:
        if not exists:
            f.write(','.join(header)+'\n')
        for mode in ('noise','gt_noise','proposal'):
            for enc in ('clean','poly'):
                row = [str(epoch), mode, enc]
                md = metrics.get((mode, enc), {})
                for k in header[3:]:
                    v = md.get(k, float('nan'))
                    row.append(f"{v:.6f}" if (isinstance(v,float) and v==v) else 'nan')
                f.write(','.join(row)+'\n')


def main() -> None:
    ap = argparse.ArgumentParser(description='Watch milestones and run compare inference')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--agg-pred-root', required=True)
    ap.add_argument('--stats-json', required=True)
    ap.add_argument('--clean-out-root', required=True)
    ap.add_argument('--poly-out-root', required=True)
    ap.add_argument('--compare-out-root', required=True)
    ap.add_argument('--polydiff-cfg', default='official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py')
    ap.add_argument('--pretrained-maptr-ckpt', default='global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth')
    ap.add_argument('--milestones', nargs='+', type=int, default=[50,100,200,400,800])
    ap.add_argument('--infer-gpu', default='6')
    ap.add_argument('--start-sigma', type=float, default=1.5)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.6)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--thr', type=float, default=0.2)
    ap.add_argument('--nms-meters', type=float, default=0.0)
    ap.add_argument('--topk', type=int, default=0)
    ap.add_argument('--min-len-norm', type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.compare_out_root, exist_ok=True)
    csv_path = osp.join(args.compare_out_root, args.scene, 'metrics.csv')
    os.makedirs(osp.join(args.compare_out_root, args.scene), exist_ok=True)

    for ep in args.milestones:
        ck_clean = osp.join(args.clean_out_root, args.scene, f'ckpt_ep_{ep:04d}.pth')
        ck_poly  = osp.join(args.poly_out_root,  args.scene, f'ckpt_ep_{ep:04d}.pth')
        done_flag = osp.join(args.compare_out_root, args.scene, f'.done_ep_{ep:04d}')
        if osp.exists(done_flag):
            continue
        print(f"[watch] waiting for ckpts epoch {ep}...\n  clean={ck_clean}\n  poly ={ck_poly}")
        while not (osp.exists(ck_clean) and osp.exists(ck_poly)):
            time.sleep(60)
        out_root = osp.join(args.compare_out_root, f'ep_{ep:04d}')
        os.makedirs(out_root, exist_ok=True)
        cmd = [
            'python','-u','global_diffusion_map/refine/clean/compare_infer_one_scene.py',
            '--static-root', args.static_root,
            '--rendered-root', args.rendered_root,
            '--agg-pred-root', args.agg_pred_root,
            '--stats-json', args.stats_json,
            '--scene', args.scene,
            '--ckpt-clean', ck_clean,
            '--ckpt-poly', ck_poly,
            '--modes','noise','gt_noise','proposal',
            '--start-sigma', str(float(args.start_sigma)),
            '--steps', str(int(args.steps)),
            '--sigma-min', str(float(args.sigma_min)),
            '--sigma-max', str(float(args.sigma_max)),
            '--rho', str(float(args.rho)),
            '--thr', str(float(args.thr)),
            '--nms-meters', str(float(args.nms_meters)),
            '--topk', str(int(args.topk)),
            '--min-len-norm', str(float(args.min_len_norm)),
            '--out-root', out_root,
            '--polydiff-cfg', args.polydiff_cfg,
            '--pretrained-maptr-ckpt', args.pretrained_maptr_ckpt,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(args.infer_gpu)
        print('[run compare]', ' '.join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
        lines: List[str] = []
        assert p.stdout is not None
        for ln in p.stdout:
            print(ln, end='')
            lines.append(ln)
        rc = p.wait()
        if rc != 0:
            print(f"[warn] compare exited with code {rc}")
        # parse metrics and append CSV
        metrics = parse_metrics_from_stream(lines)
        append_metrics_csv(csv_path, ep, metrics)
        with open(done_flag, 'w') as f:
            f.write('ok')
        print(f"[ok] epoch {ep} compare done; CSV updated: {csv_path}")


if __name__ == '__main__':
    main()

