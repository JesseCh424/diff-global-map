#!/usr/bin/env python
"""
Simple visualization of aggregated scene vectors (from export_scene_grouped.py output).
Generates merged visualization following MapTracker's official structure.

Usage:
    python viz_aggregated_simple.py \
        --scene-dir work_dirs/aggregated_scene_vectors/av2_oldsplit \
        --out-dir viz/av2_old/aggregated_pred \
        --max-scenes 10
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize aggregated scene vectors')
    parser.add_argument('--scene-dir', required=True, help='Directory with scene pickle files')
    parser.add_argument('--out-dir', required=True, help='Output directory for visualizations')
    parser.add_argument('--max-scenes', type=int, default=10, help='Maximum number of scenes to visualize')
    parser.add_argument('--dpi', type=int, default=20, help='DPI of output images')
    parser.add_argument('--line-opacity', type=float, default=0.75, help='Line opacity')
    return parser.parse_args()


def combine_images_with_labels(image_paths, labels, output_path):
    """Combine multiple images side-by-side with labels (like MapTracker)."""
    images = [cv2.imread(path) for path in image_paths]

    # Determine max dimensions
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)

    # Resize all to same size
    resized_images = []
    for img in images:
        if img.shape[0] != max_height or img.shape[1] != max_width:
            img = cv2.resize(img, (max_width, max_height))
        resized_images.append(img)

    # Concatenate horizontally
    combined = cv2.hconcat(resized_images)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 60
    for i, label in enumerate(labels):
        x_pos = int(i * max_width + max_width / 2 - len(label) * 15)
        cv2.putText(combined, label, (x_pos, y_offset), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imwrite(output_path, combined)


def visualize_scene_merged(scene_pkl_path, scene_dir, dpi=20, line_opacity=0.75):
    """
    Visualize a single scene's aggregated vectors (MERGED version).
    This matches MapTracker's 'pred_merged.png' output.
    """

    # Load scene vectors (already merged by export_scene_grouped.py)
    with open(scene_pkl_path, 'rb') as f:
        scene_data = pickle.load(f)

    scene_name = os.path.basename(scene_pkl_path).replace('.pkl', '')

    # Extract vectors by class
    crossings = scene_data.get(0, [])  # Blue
    dividers = scene_data.get(1, [])   # Red
    boundaries = scene_data.get(2, []) # Green

    # Calculate bounds
    all_points = []
    for vectors in [crossings, dividers, boundaries]:
        for vec in vectors:
            if len(vec) > 0:
                all_points.extend(vec)

    if len(all_points) == 0:
        print(f"[WARN] {scene_name}: No vectors to plot")
        return None

    all_points = np.array(all_points)
    x_min, y_min = all_points.min(axis=0) - 5
    x_max, y_max = all_points.max(axis=0) + 5

    # Create figure (matching MapTracker style)
    fig = plt.figure(figsize=(int(abs(x_min) + abs(x_max)) + 10, int(abs(y_min) + abs(y_max)) + 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Plot vectors (matching MapTracker colors: blue=crossing, red=divider, green=boundary)
    colors = {0: 'b', 1: 'r', 2: 'g'}

    for label, vectors in [(0, crossings), (1, dividers), (2, boundaries)]:
        color = colors[label]
        for vec in vectors:
            if len(vec) < 2:
                continue
            x = vec[:, 0]
            y = vec[:, 1]
            # Plot line (matching MapTracker style)
            ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=line_opacity)
            ax.plot(x, y, 'o', color=color, markersize=50)

    plt.grid(False)

    # Save merged version
    merged_path = os.path.join(scene_dir, 'pred_merged.png')
    plt.savefig(merged_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"[OK] {scene_name}: merged -> {merged_path}")
    return len(crossings), len(dividers), len(boundaries)


def main():
    args = parse_args()

    # Get scene files
    scene_files = sorted([f for f in os.listdir(args.scene_dir) if f.endswith('.pkl')])[:args.max_scenes]

    print(f"Visualizing {len(scene_files)} scenes from {args.scene_dir}")
    print(f"Output structure: {args.out_dir}/<scene_name>/pred_merged.png")

    stats = []
    for scene_file in scene_files:
        scene_pkl = os.path.join(args.scene_dir, scene_file)
        scene_name = scene_file.replace('.pkl', '')

        # Create scene directory (MapTracker structure)
        scene_dir = os.path.join(args.out_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        try:
            result = visualize_scene_merged(
                scene_pkl,
                scene_dir,
                dpi=args.dpi,
                line_opacity=args.line_opacity
            )
            if result:
                stats.append((scene_name, *result))
        except Exception as e:
            print(f"[ERROR] {scene_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Scene':<45} {'Crossing':>10} {'Divider':>10} {'Boundary':>10}")
    print("-" * 80)
    for scene_name, n_cross, n_div, n_bound in stats:
        print(f"{scene_name:<45} {n_cross:>10} {n_div:>10} {n_bound:>10}")
    print(f"\nTotal visualizations: {len(stats)}")
    print(f"Output directory structure: {args.out_dir}/<scene_name>/pred_merged.png")


if __name__ == '__main__':
    main()
