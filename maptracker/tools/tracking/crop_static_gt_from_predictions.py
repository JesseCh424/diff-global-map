#!/usr/bin/env python
"""
Crop static AV2 GT vectors by the union of prediction ROIs, optionally expanded
with aggregated prediction coverage.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mmcv
import numpy as np
from mmcv import Config
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)
from shapely.ops import unary_union

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
PKG_ROOT = TOOLS_DIR.parent
for candidate in (SCRIPT_DIR, TOOLS_DIR, PKG_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from plugin.datasets.map_utils.av2map_extractor import AV2MapExtractor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop static GT using union ROI.")
    parser.add_argument("config", help="MapTracker config file.")
    parser.add_argument(
        "--pred-pkl",
        required=True,
        help="pos_predictions_*.pkl produced by prepare_pred_tracks.py",
    )
    parser.add_argument(
        "--aggregated-pred-dir",
        default=None,
        help="Optional directory of aggregated prediction pickles (extends the ROI).",
    )
    parser.add_argument("--scene-id", required=True, help="Scene/log identifier.")
    parser.add_argument("--out-dir", required=True, help="Directory for output pickles.")
    parser.add_argument(
        "--map-info",
        default=None,
        help="Optional override for cfg.match_config.ann_file",
    )
    parser.add_argument(
        "--roi-size",
        nargs=2,
        type=float,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Override ROI size (defaults to cfg.roi_size).",
    )
    parser.add_argument(
        "--min-crossing-area",
        type=float,
        default=1.0,
        help="Minimum pedestrian crossing area (m^2).",
    )
    parser.add_argument(
        "--line-decimals",
        type=int,
        default=3,
        help="Quantization decimals for deduplication.",
    )
    return parser.parse_args()


def resolve_path(path_str: str, anchor: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.exists():
        return path
    candidate = (anchor / path_str).expanduser()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unable to resolve path: {path_str}")


def flatten_lines(geom) -> List[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [line for line in geom.geoms if not line.is_empty]
    if isinstance(geom, Polygon):
        lines = [LineString(geom.exterior.coords)]
        lines.extend(LineString(inter.coords) for inter in geom.interiors)
        return lines
    if isinstance(geom, MultiPolygon):
        lines: List[LineString] = []
        for poly in geom.geoms:
            lines.extend(flatten_lines(poly))
        return lines
    if isinstance(geom, GeometryCollection):
        lines: List[LineString] = []
        for g in geom.geoms:
            lines.extend(flatten_lines(g))
        return lines
    return []


def flatten_polygons(geom) -> List[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [poly for poly in geom.geoms if not poly.is_empty]
    if isinstance(geom, GeometryCollection):
        polys: List[Polygon] = []
        for g in geom.geoms:
            polys.extend(flatten_polygons(g))
        return polys
    return []


def canonicalize_key(coords: np.ndarray, decimals: int) -> Tuple[Tuple[float, float], ...]:
    rounded = np.round(coords, decimals=decimals)
    key_fwd = tuple(map(tuple, rounded))
    key_rev = tuple(map(tuple, rounded[::-1]))
    return key_fwd if key_fwd <= key_rev else key_rev


def collect_frames(predictions: Iterable[dict], scene_id: str) -> List[dict]:
    frames = [entry for entry in predictions if entry["scene_name"] == scene_id]
    if not frames:
        raise KeyError(f"Scene {scene_id} not found in predictions.")
    frames.sort(key=lambda item: item["local_idx"])
    return frames


def compute_roi_union_last(
    frames: Sequence[dict],
    roi_size: Sequence[float],
    last_rot2: np.ndarray,
    last_trans2: np.ndarray,
) -> Polygon:
    half_w = float(roi_size[0]) / 2.0
    half_h = float(roi_size[1]) / 2.0
    rect = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float64,
    )
    polygons = []
    for entry in frames:
        meta = entry["meta"]
        rot = np.asarray(meta["ego2global_rotation"], dtype=np.float64)[:2, :2]
        trans = np.asarray(meta["ego2global_translation"], dtype=np.float64)[:2]
        world = (rot @ rect.T).T + trans
        last_frame = (last_rot2.T @ (world - last_trans2).T).T
        polygons.append(Polygon(last_frame))
    union = unary_union(polygons)
    if union.is_empty:
        raise RuntimeError("ROI union is empty; check predictions.")
    return union


def clip_polygons(polygons: Iterable[Polygon], mask, min_area: float) -> List[np.ndarray]:
    results: List[np.ndarray] = []
    for poly in polygons:
        clipped = poly.intersection(mask)
        for piece in flatten_polygons(clipped):
            if piece.area < min_area:
                continue
            coords = np.asarray(piece.exterior.coords, dtype=np.float64)
            if coords.shape[0] < 4:
                continue
            results.append(coords[:, :2].astype(np.float32))
    return results


def clip_lines(lines: Iterable[LineString], mask, decimals: int) -> List[np.ndarray]:
    results: List[np.ndarray] = []
    seen: set = set()
    for line in lines:
        clipped = line.intersection(mask)
        for seg in flatten_lines(clipped):
            coords = np.asarray(seg.coords, dtype=np.float64)
            if coords.shape[0] < 2:
                continue
            coords = coords[:, :2]
            if np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]
            if coords.shape[0] < 2:
                continue
            key = canonicalize_key(coords, decimals)
            if key in seen:
                continue
            seen.add(key)
            results.append(coords.astype(np.float32))
    return results


def compute_car_trajectory(
    frames: Sequence[dict],
    last_rot2: np.ndarray,
    last_trans2: np.ndarray,
) -> List[dict]:
    trajectory: List[dict] = []
    for entry in frames:
        meta = entry["meta"]
        trans = np.asarray(meta["ego2global_translation"], dtype=np.float64)[:2]
        rot = np.asarray(meta["ego2global_rotation"], dtype=np.float64)[:2, :2]
        center = last_rot2.T @ (trans - last_trans2)
        rot_rel = last_rot2.T @ rot
        yaw_deg = math.degrees(math.atan2(rot_rel[1, 0], rot_rel[0, 0]))
        frame_idx = int(meta.get("sample_idx", entry.get("local_idx", 0)))
        trajectory.append(
            {
                "frame_idx": frame_idx,
                "center": center.astype(np.float32).tolist(),
                "yaw_deg": float(yaw_deg),
            }
        )
    trajectory.sort(key=lambda item: item["frame_idx"])
    return trajectory


def load_aggregated_mask(agg_dir: Optional[Path], scene: str) -> Optional[Polygon]:
    if agg_dir is None:
        return None
    agg_path = agg_dir / f"{scene}.pkl"
    if not agg_path.exists():
        return None
    data = mmcv.load(str(agg_path))
    shapes = []
    for arr in data.get(0, []):
        if len(arr) >= 3:
            shapes.append(Polygon(arr))
    for lbl in (1, 2):
        for arr in data.get(lbl, []):
            if len(arr) >= 2:
                line = LineString(arr)
                shapes.append(line.buffer(1.0, cap_style=2, join_style=2))
    if not shapes:
        return None
    try:
        return unary_union(shapes)
    except Exception as exc:
        print(f"[warn] aggregated mask union failed for {scene}: {exc}")
        return None


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)

    roi_size = args.roi_size or cfg.get("roi_size")
    if roi_size is None:
        raise ValueError("ROI size not provided and missing in config.")

    map_info_path = args.map_info or cfg.match_config.get("ann_file")
    if map_info_path is None:
        raise ValueError("map info path not provided and missing in config.")

    cfg_dir = Path(args.config).resolve().parent
    map_info = resolve_path(map_info_path, cfg_dir)
    id2map = mmcv.load(str(map_info))["id2map"]

    scene_id = args.scene_id
    if scene_id not in id2map:
        raise KeyError(f"Scene {scene_id} not present in map info.")
    map_json = Path(id2map[scene_id])
    if not map_json.exists():
        raise FileNotFoundError(f"Static map JSON missing: {map_json}")

    predictions = mmcv.load(args.pred_pkl)
    frames = collect_frames(predictions, scene_id)
    last_meta = frames[-1]["meta"]
    last_rot_full = np.asarray(last_meta["ego2global_rotation"], dtype=np.float64)
    last_trans_full = np.asarray(last_meta["ego2global_translation"], dtype=np.float64)
    last_rot2 = last_rot_full[:2, :2]
    last_trans2 = last_trans_full[:2]

    roi_union = compute_roi_union_last(frames, roi_size, last_rot2, last_trans2)
    agg_mask = load_aggregated_mask(
        Path(args.aggregated_pred_dir) if args.aggregated_pred_dir else None,
        scene_id,
    )
    if agg_mask is not None:
        combined_union = unary_union([roi_union, agg_mask])
    else:
        combined_union = roi_union

    # Shapely returns bounds as (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = combined_union.bounds
    extent_x = max(abs(minx), abs(maxx))
    extent_y = max(abs(miny), abs(maxy))
    extractor_roi = (
        max(float(roi_size[0]), 2.0 * extent_x + 20.0),
        max(float(roi_size[1]), 2.0 * extent_y + 20.0),
    )

    extractor = AV2MapExtractor(extractor_roi, {scene_id: str(map_json)})
    map_geom = extractor.get_map_geom(scene_id, last_trans_full, last_rot_full)

    ped_shapes = []
    for geom in map_geom.get("ped_crossing", []):
        if isinstance(geom, Polygon):
            ped_shapes.append(geom)
        else:
            coords = list(geom.coords)
            if len(coords) >= 4 and np.allclose(coords[0], coords[-1]):
                ped_shapes.append(Polygon(coords))
    ped_vecs = clip_polygons(ped_shapes, combined_union, args.min_crossing_area)
    divider_vecs = clip_lines(map_geom.get("divider", []), combined_union, args.line_decimals)
    boundary_vecs = clip_lines(map_geom.get("boundary", []), combined_union, args.line_decimals)
    car_traj = compute_car_trajectory(frames, last_rot2, last_trans2)

    scene_bank: Dict[str, List[np.ndarray]] = {
        0: ped_vecs,
        1: divider_vecs,
        2: boundary_vecs,
        "car_trajectory": car_traj,
        "bounds": [minx, miny, maxx, maxy],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_id}.pkl"
    mmcv.dump(scene_bank, str(out_path))
    print(f"[ok] static GT saved: {out_path} bounds=[{minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f}]")


if __name__ == "__main__":
    main()
