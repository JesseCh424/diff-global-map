#!/usr/bin/env python
from __future__ import annotations

"""
Fix 10_render_gt.png alignment by re-rendering with canonical static-GT bounds
and the semantic (08) canvas size.

Policy
- Bounds: strictly read from static GT pkl (key 'bounds' = [minx,miny,maxx,maxy]).
- Canvas size: reuse 08_agg_semantic.png size under the semantic root.
- Only re-render scenes whose existing 10 size differs from 08 size, or when 10 is missing.

Usage example:
  python -u global_diffusion_map/refine/tools/fix_condition_canvases.py \
    --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
    --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
    --aggregated-pred maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid \
    --aggregated-gt   maptracker/work_dirs/agg_gt_vector/av2_oldsplit/val \
    --semantic-root   maptracker/work_dirs/semantic/valid \
    --scenes <ID ...>
"""

import argparse
import os
import os.path as osp
import subprocess
from typing import List, Optional, Tuple

from PIL import Image


def img_size(path: str) -> Optional[Tuple[int, int]]:
    try:
        im = Image.open(path)
        return im.size  # (W,H)
    except Exception:
        return None


def sem_canvas_size(sem_root: str, scene: str) -> Optional[Tuple[int, int]]:
    p = osp.join(sem_root, scene, '08_agg_semantic.png')
    return img_size(p)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Fix condition canvases (10_render_gt.png) to match static bounds and 08 size')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--aggregated-pred', required=True)
    ap.add_argument('--aggregated-gt', required=True)
    ap.add_argument('--semantic-root', required=True)
    ap.add_argument('--scenes', nargs='*', default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenes is None or len(args.scenes) == 0:
        # derive scenes from static root
        scenes = [osp.splitext(f)[0] for f in os.listdir(args.static_root) if f.endswith('.pkl')]
    else:
        scenes = list(args.scenes)

    fixed: List[str] = []
    skipped: List[str] = []
    errors: List[str] = []

    for s in scenes:
        sem_sz = sem_canvas_size(args.semantic_root, s)
        if sem_sz is None:
            skipped.append(f"{s}: missing 08_agg_semantic.png under {args.semantic_root}")
            continue
        p10 = osp.join(args.rendered_root, s, '10_render_gt.png')
        ten_sz = img_size(p10)
        need = (ten_sz is None) or (ten_sz != sem_sz)
        if not need:
            skipped.append(f"{s}: 10 OK (size={ten_sz}) matches 08 {sem_sz}")
            continue
        try:
            # Re-render this single scene by invoking the canonical script for just this scene
            cmd = [
                'python', '-u', 'maptracker/tools/tracking/render_gt_to_10.py',
                '--static-root', args.static_root,
                '--aggregated-pred', args.aggregated_pred,
                '--aggregated-gt', args.aggregated_gt,
                '--semantic-root', args.semantic_root,
                '--out-root', args.rendered_root,
                '--scenes', s
            ]
            subprocess.check_call(cmd)
            fixed.append(f"{s}: re-rendered 10 to match 08 size {sem_sz}")
        except subprocess.CalledProcessError as e:
            errors.append(f"{s}: render_gt_to_10 failed: {e}")
        except Exception as e:
            errors.append(f"{s}: unexpected error: {e}")

    print('[fix] done')
    print('[fix] fixed:')
    for m in fixed:
        print('  -', m)
    print('[fix] skipped:')
    for m in skipped:
        print('  -', m)
    if errors:
        print('[fix] errors:')
        for m in errors:
            print('  -', m)


if __name__ == '__main__':
    main()

