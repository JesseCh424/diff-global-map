#!/usr/bin/env python
"""
Standalone visualization using official MapTracker vis_global.py functions.
This bypasses the dataset requirement and directly visualizes pos_predictions_5.pkl
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse
import pickle
import numpy as np
import torch
from collections import defaultdict
from mmcv import Config

# Import official MapTracker visualization functions
from vis_global import (
    plot_fig_merged,
    plot_fig_unmerged,
    combine_images_with_labels,
    get_prev2curr_vectors
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--data-path', required=True, help='Path to pos_predictions_5.pkl')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--scene-id', type=str, nargs='+', default=None, help='Scene IDs to visualize')
    parser.add_argument('--simplify', type=float, default=0.5, help='Simplification tolerance')
    parser.add_argument('--dpi', type=int, default=20, help='DPI')
    parser.add_argument('--line-opacity', type=float, default=0.75, help='Line opacity')
    parser.add_argument('--transparent', action='store_true', help='Transparent background')
    parser.add_argument('--overwrite', type=int, default=1, help='Overwrite existing')
    return parser.parse_args()


class Args:
    """Mimic args object for vis_global functions"""
    def __init__(self, simplify, dpi, line_opacity, transparent):
        self.simplify = simplify
        self.dpi = dpi
        self.line_opacity = line_opacity
        self.transparent = transparent


def vis_pred_official(scene_name, pred_results, origin, roi_size, args, out_dir):
    """
    Use official MapTracker visualization logic.
    Adapted from vis_global.py:vis_pred_data()
    """

    # Get frames for this scene
    index_list = []
    for index in range(len(pred_results)):
        if pred_results[index]["scene_name"] == scene_name:
            index_list.append(index)

    if not index_list:
        print(f"[SKIP] {scene_name}: No frames found")
        return

    car_trajectory = []
    id_prev2curr_pred_vectors = defaultdict(list)
    id_prev2curr_pred_frame_info = defaultdict(list)
    id_prev2curr_pred_frame = defaultdict(list)

    # Iterate through each frame
    last_index = index_list[-1]
    for local_idx, index in enumerate(index_list):

        vectors = np.array(pred_results[index]["vectors"]).reshape((len(np.array(pred_results[index]["vectors"])), 20, 2))
        if abs(vectors.max()) <= 1:
            curr_vectors = vectors * roi_size + origin
        else:
            curr_vectors = vectors

        # Get transformation matrix to last frame
        prev_e2g_trans = torch.tensor(pred_results[index]['meta']['ego2global_translation'], dtype=torch.float64)
        prev_e2g_rot = torch.tensor(pred_results[index]['meta']['ego2global_rotation'], dtype=torch.float64)
        curr_e2g_trans = torch.tensor(pred_results[last_index]['meta']['ego2global_translation'], dtype=torch.float64)
        curr_e2g_rot = torch.tensor(pred_results[last_index]['meta']['ego2global_rotation'], dtype=torch.float64)

        prev_e2g_matrix = torch.eye(4, dtype=torch.float64)
        prev_e2g_matrix[:3, :3] = prev_e2g_rot
        prev_e2g_matrix[:3, 3] = prev_e2g_trans

        curr_g2e_matrix = torch.eye(4, dtype=torch.float64)
        curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
        curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

        prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix

        # Transform vectors to last frame coordinate
        prev2curr_vecs = get_prev2curr_vectors(
            curr_vectors,
            prev2curr_matrix,
            origin,
            roi_size,
            denormalize=False,
            clip=False
        )
        prev2curr_vecs = prev2curr_vecs * roi_size + origin

        # Store by global_id
        for i, (label, global_id) in enumerate(zip(pred_results[index]["labels"], pred_results[index]["global_ids"])):
            dict_key = f"{int(label)}_{int(global_id)}"
            id_prev2curr_pred_vectors[dict_key].append(prev2curr_vecs[i].numpy())
            id_prev2curr_pred_frame_info[dict_key].append([local_idx, len(id_prev2curr_pred_frame[dict_key])])

        # Car trajectory
        rotation_degrees = np.degrees(np.arctan2(prev2curr_matrix[:3, :3][1, 0], prev2curr_matrix[:3, :3][0, 0]))
        car_center = get_prev2curr_vectors(
            np.array((0, 0)).reshape(1, 1, 2),
            prev2curr_matrix,
            origin,
            roi_size,
            False,
            False
        ) * roi_size + origin
        car_trajectory.append([car_center.squeeze(), rotation_degrees])

    # Sort vectors
    id_prev2curr_pred_vectors = {key: id_prev2curr_pred_vectors[key] for key in sorted(id_prev2curr_pred_vectors)}

    # Calculate bounds
    x_min = -roi_size[0] / 2
    x_max = roi_size[0] / 2
    y_min = -roi_size[1] / 2
    y_max = roi_size[1] / 2

    all_points = []
    for vecs in id_prev2curr_pred_vectors.values():
        points = np.concatenate(vecs, axis=0)
        all_points.append(points)

    if all_points:
        all_points = np.concatenate(all_points, axis=0)
        x_min = min(x_min, all_points[:, 0].min())
        x_max = max(x_max, all_points[:, 0].max())
        y_min = min(y_min, all_points[:, 1].min())
        y_max = max(y_max, all_points[:, 1].max())

    # Create scene directory
    scene_dir = os.path.join(out_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    # Plot using official functions
    pred_save_path = os.path.join(scene_dir, 'pred_unmerged.png')
    plot_fig_unmerged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, id_prev2curr_pred_vectors, args)

    pred_save_path = os.path.join(scene_dir, 'pred_merged.png')
    plot_fig_merged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, id_prev2curr_pred_vectors, args)

    comb_save_path = os.path.join(scene_dir, 'pred_comb.png')
    image_paths = [os.path.join(scene_dir, 'pred_merged.png'), os.path.join(scene_dir, 'pred_unmerged.png')]
    labels = ['Merged', 'Unmerged']
    combine_images_with_labels(image_paths, labels, comb_save_path)

    print(f"[OK] {scene_name}: Official MapTracker visualization saved to {scene_dir}")


def main():
    args_parsed = parse_args()
    cfg = Config.fromfile(args_parsed.config)

    roi_size = torch.tensor(cfg.roi_size).numpy()
    origin = torch.tensor(cfg.pc_range[:2]).numpy()

    # Load predictions
    with open(args_parsed.data_path, 'rb') as f:
        pred_results = pickle.load(f)

    # Get all scenes
    all_scenes = sorted(list(set([p["scene_name"] for p in pred_results])))

    # Filter by scene_id if specified
    if args_parsed.scene_id:
        all_scenes = [s for s in all_scenes if s in args_parsed.scene_id]

    print(f"Visualizing {len(all_scenes)} scenes using official MapTracker functions")

    # Create args object for vis functions
    vis_args = Args(args_parsed.simplify, args_parsed.dpi, args_parsed.line_opacity, args_parsed.transparent)

    for scene_name in all_scenes:
        scene_dir = os.path.join(args_parsed.out_dir, scene_name)
        if os.path.exists(scene_dir) and not args_parsed.overwrite:
            print(f"[SKIP] {scene_name}: Already exists")
            continue

        vis_pred_official(scene_name, pred_results, origin, roi_size, vis_args, args_parsed.out_dir)


if __name__ == '__main__':
    main()
