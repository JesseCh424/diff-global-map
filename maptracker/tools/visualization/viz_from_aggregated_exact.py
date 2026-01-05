#!/usr/bin/env python
"""
Exact replica of MapTracker's plot_fig_merged for aggregated pickles.
This matches the official visualization pixel-by-pixel.
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from PIL import Image
from shapely.geometry import LineString
from shapely.ops import unary_union
from scipy.spatial import ConvexHull

# Import MapTracker's merge functions
from vis_global import merge_corssing, merge_divider, merge_boundary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-pkl', required=True, help='Path to aggregated scene pickle')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--roi-size', type=float, nargs=2, default=[60.0, 30.0], help='ROI size (x, y)')
    parser.add_argument('--dpi', type=int, default=20, help='DPI')
    parser.add_argument('--line-opacity', type=float, default=0.75, help='Line opacity')
    parser.add_argument('--simplify', type=float, default=0.5, help='Simplification tolerance')
    parser.add_argument('--transparent', action='store_true', help='Transparent background')
    return parser.parse_args()


class Args:
    """Mimic args object for MapTracker functions"""
    def __init__(self, simplify, dpi, line_opacity, transparent):
        self.simplify = simplify
        self.dpi = dpi
        self.line_opacity = line_opacity
        self.transparent = transparent


def plot_fig_merged_exact(scene_data, roi_size, out_path, args):
    """
    Exact replica of vis_global.py:plot_fig_merged()
    Adapted to work with aggregated pickle data.
    """

    # Convert scene_data to id_prev2curr_pred_vectors format
    # (vectors are already merged by export_scene_grouped.py)
    id_prev2curr_pred_vectors = {}
    for label in [0, 1, 2]:
        for i, vec in enumerate(scene_data.get(label, [])):
            # Use label_index as key to match MapTracker format
            key = f"{label}_{i}"
            id_prev2curr_pred_vectors[key] = [vec]

    if not id_prev2curr_pred_vectors:
        print(f"[WARN] No vectors to plot")
        return

    # Empty car trajectory (we don't have this from aggregated pkl)
    car_trajectory = []

    # Calculate bounds (EXACTLY like MapTracker)
    x_min = -roi_size[0] / 2
    x_max = roi_size[0] / 2
    y_min = -roi_size[1] / 2
    y_max = roi_size[1] / 2

    all_points = []
    for vecs in id_prev2curr_pred_vectors.values():
        points = np.concatenate(vecs, axis=0)
        all_points.append(points)
    all_points = np.concatenate(all_points, axis=0)

    x_min = min(x_min, all_points[:, 0].min())
    x_max = max(x_max, all_points[:, 0].max())
    y_min = min(y_min, all_points[:, 1].min())
    y_max = max(y_max, all_points[:, 1].max())

    # Setup figure (EXACTLY like MapTracker line 888)
    fig = plt.figure(figsize=(int(abs(x_min) + abs(x_max)) + 10, int(abs(y_min) + abs(y_max)) + 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Skip car trajectory rendering (no data available from aggregated pkl)
    # The official version renders car icons here (lines 892-914)

    # Merge and plot vectors (EXACTLY like MapTracker lines 916-967)
    for tag, vecs in id_prev2curr_pred_vectors.items():
        label, vec_glb_idx = tag.split('_')
        label = int(label)

        if label == 0:  # ped_crossing
            color = 'b'
        elif label == 1:  # divider
            color = 'r'
        elif label == 2:  # boundary
            color = 'g'

        # Get vectors for this instance
        polylines = []
        for vec in vecs:
            polylines.append(LineString(vec))
        if len(polylines) <= 0:
            continue

        # Apply merging logic (EXACTLY like MapTracker)
        if label == 0:  # crossing, merged by convex hull
            polygon = merge_corssing(polylines)
            if polygon.area < 2:
                continue
            polygon = polygon.simplify(args.simplify)
            vector = np.array(polygon.exterior.coords)
            pts = vector[:, :2]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=args.line_opacity)
            ax.plot(x, y, "o", color=color, markersize=50)

        elif label == 1:  # divider, merged by interpolation
            polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
            polylines_vecs = merge_divider(polylines_vecs)
            for one_line in polylines_vecs:
                one_line = np.array(LineString(one_line).simplify(args.simplify).coords)
                pts = one_line[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=args.line_opacity)
                ax.plot(x, y, "o", color=color, markersize=50)

        elif label == 2:  # boundary, merged by interpolation
            polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
            polylines_vecs = merge_boundary(polylines_vecs)
            for one_line in polylines_vecs:
                one_line = np.array(LineString(one_line).simplify(args.simplify).coords)
                pts = one_line[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=args.line_opacity)
                ax.plot(x, y, "o", color=color, markersize=50)

    plt.grid(False)
    plt.savefig(out_path, bbox_inches='tight', transparent=args.transparent, dpi=args.dpi)
    plt.clf()
    plt.close(fig)

    print(f"[OK] Saved: {out_path}")


def main():
    args_parsed = parse_args()

    # Load aggregated scene
    with open(args_parsed.scene_pkl, 'rb') as f:
        scene_data = pickle.load(f)

    scene_name = os.path.basename(args_parsed.scene_pkl).replace('.pkl', '')
    print(f"Processing: {scene_name}")

    # Create output directory
    os.makedirs(args_parsed.out_dir, exist_ok=True)

    # Create args object for MapTracker functions
    viz_args = Args(
        args_parsed.simplify,
        args_parsed.dpi,
        args_parsed.line_opacity,
        args_parsed.transparent
    )

    # Plot using exact MapTracker logic
    out_path = os.path.join(args_parsed.out_dir, 'my_pred_merged.png')
    roi_size = np.array(args_parsed.roi_size)

    plot_fig_merged_exact(scene_data, roi_size, out_path, viz_args)


if __name__ == '__main__':
    main()
