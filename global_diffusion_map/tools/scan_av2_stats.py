#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import numpy as np
from typing import Dict, List


def load_pkl(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def percentile(arr: List[int], q: float) -> float:
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(np.asarray(arr, dtype=np.float64), q * 100.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--static-root', required=True, help='Path to simplified static GT pkls (train)')
    ap.add_argument('--out-json', required=True, help='Where to write the computed stats JSON')
    ap.add_argument('--vertex-quantile', type=float, default=0.95)
    ap.add_argument('--inst-quantile', type=float, default=0.98)
    ap.add_argument('--min-M', type=int, default=24)
    ap.add_argument('--max-M', type=int, default=64)
    ap.add_argument('--max-num-queries', type=int, default=1024)
    args = ap.parse_args()

    files = [p for p in os.listdir(args.static_root) if p.endswith('.pkl')]
    if not files:
        raise SystemExit(f'No .pkl files under {args.static_root}')

    vertex_counts: List[int] = []
    per_scene_counts: Dict[int, List[int]] = {0: [], 1: [], 2: []}

    for fname in files:
        data = load_pkl(osp.join(args.static_root, fname))
        scene_counts = {0: 0, 1: 0, 2: 0}
        for cls_id in (0, 1, 2):
            vecs = data.get(cls_id, [])
            scene_counts[cls_id] = len(vecs)
            for arr in vecs:
                vertex_counts.append(int(getattr(arr, 'shape', [0])[0]))
        for cls_id in (0, 1, 2):
            per_scene_counts[cls_id].append(scene_counts[cls_id])

    # Compute M by vertex distribution
    vq = percentile(vertex_counts, args.vertex_quantile)
    M = int(max(args.min_M, min(args.max_M, int(np.ceil(vq)))))

    # Class budgets by per-scene instance counts
    budgets: Dict[int, int] = {}
    for cls_id in (0, 1, 2):
        q = percentile(per_scene_counts[cls_id], args.inst_quantile)
        budgets[cls_id] = int(max(1, int(np.ceil(q))))

    num_queries = int(sum(budgets.values()))
    if num_queries > args.max_num_queries:
        # scale down budgets proportionally
        scale = args.max_num_queries / float(num_queries)
        for k in budgets.keys():
            budgets[k] = max(1, int(np.floor(budgets[k] * scale)))
        num_queries = int(sum(budgets.values()))

    out = {
        'M': M,
        'num_queries': num_queries,
        'class_budget': budgets,
        'vertex_quantile': args.vertex_quantile,
        'inst_quantile': args.inst_quantile,
        'static_root': args.static_root,
    }
    os.makedirs(osp.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

