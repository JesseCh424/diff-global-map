#!/usr/bin/env python
"""Add bounds key to existing static GT pickles by computing from vector data."""

import argparse
import glob
from pathlib import Path
import mmcv
import numpy as np


def compute_bounds_from_vectors(data):
    """Compute bounds [minx, miny, maxx, maxy] from all vectors in the scene."""
    all_pts = []
    for cls_id in [0, 1, 2]:
        for vec in data.get(cls_id, []):
            arr = np.asarray(vec, dtype=np.float32)
            if len(arr) > 0:
                all_pts.append(arr)

    if not all_pts:
        return [0, 0, 1, 1]

    cat = np.concatenate(all_pts, axis=0)
    minx, miny = cat.min(axis=0)
    maxx, maxy = cat.max(axis=0)
    return [float(minx), float(miny), float(maxx), float(maxy)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl-root', required=True, help='Directory containing pickles')
    parser.add_argument('--pattern', default='*.pkl', help='Glob pattern for pickles')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without modifying files')
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(f"{args.pkl_root}/{args.pattern}"))
    print(f"Found {len(pkl_files)} pickle files")

    updated = 0
    skipped = 0

    for pkl_path in pkl_files:
        data = mmcv.load(pkl_path)

        if 'bounds' in data:
            skipped += 1
            continue

        bounds = compute_bounds_from_vectors(data)

        if args.dry_run:
            print(f"[dry-run] Would add bounds={bounds} to {Path(pkl_path).name}")
        else:
            data['bounds'] = bounds
            mmcv.dump(data, pkl_path)
            print(f"[updated] {Path(pkl_path).name}: bounds={bounds}")

        updated += 1

    print(f"\nSummary: {updated} updated, {skipped} already had bounds")


if __name__ == '__main__':
    main()
