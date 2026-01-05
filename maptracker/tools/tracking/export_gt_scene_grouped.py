#!/usr/bin/env python
"""
Aggregate ground-truth scene vectors into per-scene pickles, mirroring the
prediction aggregator (export_scene_grouped.py) but using gt tracks.

Inputs:
- config: MapTracker config (for dataset + roi_size/pc_range)
- --gt-tracks: path to *_gt_tracks.pkl produced by prepare_gt_tracks.py
- --out-dir: directory to write <scene>.pkl files

Each output pickle matches the prediction format:
{
    0: [np.ndarray],  # crossings in last-frame ego coordinates
    1: [...],         # dividers
    2: [...],         # boundaries
    "car_trajectory": [
        {"frame_idx": int, "center": [x, y], "yaw_deg": float},
        ...
    ]
}
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import mmcv
import numpy as np
import torch
from mmcv import Config

# Ensure plugin modules are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(SCRIPT_DIR)
PKG_ROOT = os.path.dirname(TOOLS_DIR)
if TOOLS_DIR not in sys.path:
    sys.path.append(TOOLS_DIR)
if PKG_ROOT not in sys.path:
    sys.path.append(PKG_ROOT)

from visualization.vis_global import (  # noqa: E402
    get_consecutive_vectors_with_opt,
    get_prev2curr_vectors,
    merge_corssing,
    merge_divider,
    merge_boundary,
)
from tracking.cmap_utils.match_utils import get_prev2curr_matrix  # noqa: E402
from shapely.geometry import LineString as ShpLineString  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate MapTracker ground-truth vectors per scene."
    )
    parser.add_argument("config", help="MapTracker config file.")
    parser.add_argument(
        "--gt-tracks",
        required=True,
        help="Path to *_gt_tracks.pkl produced by prepare_gt_tracks.py",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to store aggregated GT pickles.",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.5,
        help="Simplification tolerance (meters).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scene pickles.",
    )
    return parser.parse_args()


def import_plugins(cfg: Config) -> None:
    """
    Dynamically import plugin modules specified in the config.
    """
    if not getattr(cfg, "plugin", False):
        return

    plugin_dirs = cfg.plugin_dir
    if not isinstance(plugin_dirs, list):
        plugin_dirs = [plugin_dirs]

    for plugin_dir in plugin_dirs:
        module_path = plugin_dir.rstrip("/").replace("/", ".")
        if module_path:
            __import__(module_path)


def build_dataset(cfg: Config):
    from mmdet3d.datasets import build_dataset

    dataset_cfg = cfg.match_config.copy()
    # Ensure we load the validation split (default for aggregation)
    if "train" in dataset_cfg.ann_file:
        dataset_cfg.ann_file = dataset_cfg.ann_file.replace("train", "val")
    return build_dataset(dataset_cfg)


def fetch_sample(dataset, idx: int) -> Tuple[Dict[int, List[np.ndarray]], Dict[str, Any]]:
    """
    Retrieve a dataset sample and convert vectors/meta to numpy-friendly dicts.
    """
    sample = dataset[idx]

    # Vectors: dict[label] -> list[np.ndarray] (shape: (num_pts, 2))
    raw_vectors = sample["vectors"].data
    vectors: Dict[int, List[np.ndarray]] = {}
    for label, vec_list in raw_vectors.items():
        vectors[label] = [np.asarray(v, dtype=np.float32) for v in vec_list]

    meta_dc = sample["img_metas"].data
    meta = {
        "scene_name": meta_dc["scene_name"],
        "sample_idx": int(meta_dc["sample_idx"]),
        "token": meta_dc["token"],
        "ego2global_translation": np.asarray(meta_dc["ego2global_translation"], dtype=np.float64),
        "ego2global_rotation": np.asarray(meta_dc["ego2global_rotation"], dtype=np.float64),
    }
    return vectors, meta


def aggregate_scene(
    scene: str,
    scene_track: Dict[str, Any],
    dataset,
    roi_size: np.ndarray,
    origin: np.ndarray,
    simplify_tol: float,
    overwrite: bool,
    out_dir: str,
) -> None:
    out_path = os.path.join(out_dir, f"{scene}.pkl")
    if os.path.exists(out_path) and not overwrite:
        print(f"[skip] {scene}: exists -> {out_path}")
        return

    idxs: List[int] = scene_track["sample_ids"]
    inst_seq: List[Dict[int, Dict[int, int]]] = scene_track["instance_ids"]
    if not idxs:
        print(f"[warn] {scene}: no frames")
        return

    last_idx = idxs[-1]
    _, last_meta = fetch_sample(dataset, last_idx)

    bank: Dict[str, List[np.ndarray]] = defaultdict(list)
    car_trajectory: List[Dict[str, Any]] = []

    origin_tensor = torch.tensor(origin, dtype=torch.float32)
    roi_size_tensor = torch.tensor(roi_size, dtype=torch.float32)

    for frame_pos, ds_idx in enumerate(idxs):
        vectors, meta = fetch_sample(dataset, ds_idx)
        prev_meta = {
            "ego2global_translation": meta["ego2global_translation"],
            "ego2global_rotation": meta["ego2global_rotation"],
        }
        curr_meta = {
            "ego2global_translation": last_meta["ego2global_translation"],
            "ego2global_rotation": last_meta["ego2global_rotation"],
        }
        prev2curr = get_prev2curr_matrix(prev_meta, curr_meta)
        prev2curr_np = prev2curr.detach().cpu().numpy()

        # car trajectory in last frame coordinates
        car_vec = get_prev2curr_vectors(
            np.array((0.0, 0.0), dtype=np.float32).reshape(1, 1, 2),
            prev2curr,
            origin_tensor,
            roi_size_tensor,
            denormalize=False,
            clip=False,
        )
        car_center = (car_vec * roi_size_tensor + origin_tensor).detach().cpu().numpy()[0, 0]
        center_xy = car_center.astype(np.float32)
        yaw_deg = float(np.degrees(np.arctan2(prev2curr_np[1, 0], prev2curr_np[0, 0])))
        car_trajectory.append(
            {
                "frame_idx": int(meta["sample_idx"]),
                "center": center_xy.tolist(),
                "yaw_deg": yaw_deg,
            }
        )

        label_ids_map = inst_seq[frame_pos]
        # Denormalize vectors to metric coordinates (matching vis_global)
        curr_vectors = {}
        for label in (0, 1, 2):
            vec_list = vectors.get(label, [])
            if vec_list:
                stacked = torch.tensor(np.stack(vec_list, axis=0), dtype=torch.float32)
                curr_vectors[label] = stacked * roi_size_tensor + origin_tensor
            else:
                curr_vectors[label] = []

        prev2curr = get_prev2curr_matrix(prev_meta, curr_meta)
        transformed = get_consecutive_vectors_with_opt(
            curr_vectors,
            prev2curr,
            origin_tensor,
            roi_size_tensor,
            False,
            False,
        )

        for label in (0, 1, 2):
            gid_map = label_ids_map.get(label, {})
            seq = transformed.get(label, [])
            for local_idx, gid in gid_map.items():
                vec_tensor = seq[local_idx]
                if isinstance(vec_tensor, torch.Tensor):
                    vec_abs = vec_tensor.detach().cpu().numpy()
                else:
                    vec_abs = np.asarray(vec_tensor, dtype=np.float32)
                key = f"{label}_{gid}"
                bank[key].append(vec_abs)

    merged_by_label: Dict[Any, List[np.ndarray]] = {0: [], 1: [], 2: []}
    for key, lines in bank.items():
        lbl, gid = map(int, key.split("_"))
        if not lines:
            continue
        if lbl == 0:
            polygons = [ShpLineString(v) for v in lines if len(v) >= 3]
            if polygons:
                hull = merge_corssing(polygons)
                if hull and not hull.is_empty:
                    merged_by_label[0].append(np.asarray(hull.exterior.coords))
        elif lbl == 1:
            valid = []
            for v in lines:
                arr = np.asarray(v)
                if arr.ndim != 2 or arr.shape[0] < 2:
                    continue
                valid.append(arr)
            if not valid:
                continue
            try:
                merged = merge_divider(valid)
            except ValueError:
                continue
            merged_by_label[1].extend([
                arr for arr in (np.asarray(v) for v in merged)
                if arr.ndim == 2 and arr.shape[0] >= 2
            ])
        elif lbl == 2:
            valid = []
            for v in lines:
                arr = np.asarray(v)
                if arr.ndim != 2 or arr.shape[0] < 2:
                    continue
                valid.append(arr)
            if not valid:
                continue
            try:
                merged = merge_boundary(valid)
            except ValueError:
                continue
            merged_by_label[2].extend([
                arr for arr in (np.asarray(v) for v in merged)
                if arr.ndim == 2 and arr.shape[0] >= 2
            ])

    if simplify_tol > 0:
        def _simplify(arr: np.ndarray) -> np.ndarray:
            if len(arr) < 2:
                return arr
            simp = ShpLineString(arr).simplify(simplify_tol)
            coords = np.asarray(simp.coords)
            return coords if len(coords) >= 2 else arr

        for lbl in merged_by_label:
            if lbl in (0, 1, 2):
                merged_by_label[lbl] = [_simplify(vec) for vec in merged_by_label[lbl]]

    merged_by_label["car_trajectory"] = sorted(
        car_trajectory,
        key=lambda entry: entry["frame_idx"],
    )
    mmcv.dump(merged_by_label, out_path)
    print(f"[ok] {scene}: {out_path}")


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugins(cfg)

    roi_size = np.array(cfg.get("roi_size", (60, 30)), dtype=np.float32)
    origin = np.array(
        cfg.get(
            "pc_range",
            [
                -roi_size[0] / 2,
                -roi_size[1] / 2,
                -3,
                roi_size[0] / 2,
                roi_size[1] / 2,
                5,
            ],
        )[:2],
        dtype=np.float32,
    )

    dataset = build_dataset(cfg)
    gt_tracks = mmcv.load(args.gt_tracks)

    os.makedirs(args.out_dir, exist_ok=True)

    for idx, scene in enumerate(sorted(gt_tracks.keys()), 1):
        scene_track = gt_tracks[scene]
        aggregate_scene(
            scene,
            scene_track,
            dataset,
            roi_size,
            origin,
            args.simplify,
            args.overwrite,
            args.out_dir,
        )
        if idx % 10 == 0:
            print(f"[progress] {idx}/{len(gt_tracks)} scenes processed.")


if __name__ == "__main__":
    main()
