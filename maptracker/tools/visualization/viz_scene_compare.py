#!/usr/bin/env python
"""
Generate comparison visualizations per scene:
 - 01_agg_pred_direct.png  (aggregated prediction vectors)
 - 04_agg_gt_direct.png    (aggregated ground-truth vectors)
 - 05_static_gt.png        (union-ROI cropped static GT)

All renders share the same axis limits (derived from aggregated pred/gt) and
reuse MapTracker's plot_fig_unmerged styling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
PKG_ROOT = TOOLS_DIR.parent
for candidate in (TOOLS_DIR, PKG_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from typing import Dict, List, Sequence, Tuple

import mmcv
import numpy as np

from tools.visualization.render_static_maptracker_style import (
    _Args,
    compute_bounds,
    compute_bounds_from_pkls,
    render_static_mt_style,
    to_maptracker_bank,
)
from tools.visualization.vis_global import plot_fig_unmerged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render aggregated/static comparison PNGs.")
    parser.add_argument("--aggregated-pred", required=True, help="Directory of aggregated prediction pickles.")
    parser.add_argument("--aggregated-gt", required=True, help="Directory of aggregated GT pickles.")
    parser.add_argument("--static-gt", required=True, help="Directory of static GT pickles (cropped).")
    parser.add_argument("--out-root", required=True, help="Output root (per-scene folders created here).")
    parser.add_argument("--scene-list", help="Optional text file with scene IDs (one per line).")
    parser.add_argument(
        "--bounds-pkl",
        nargs="+",
        help="Optional pickle templates (use {scene} placeholder) to derive shared axis bounds.",
    )
    parser.add_argument("--dpi", type=int, default=40, help="DPI for output images.")
    parser.add_argument("--simplify", type=float, default=0.5, help="Line simplification tolerance.")
    parser.add_argument("--line-opacity", type=float, default=0.75, help="Polyline opacity.")
    # Selective generation: only generate specified panels
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["01", "04", "05", "08"],
        default=None,
        help="If provided, generate only the listed panels (01/04/05/08). When omitted, generates 01+04+05 (08 optional via flags).",
    )
    # Optional: also generate or copy 08_agg_semantic.png
    parser.add_argument("--gen-semantic", action="store_true", help="Also generate 08_agg_semantic.png under each scene folder.")
    parser.add_argument("--sem-config", default="maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py", help="MapTracker config for semantic aggregator.")
    parser.add_argument("--submission-json", default="maptracker/work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/submission_vector.json", help="submission_vector.json path.")
    parser.add_argument("--pos-pkl", default="maptracker/work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions_5.pkl", help="pos_predictions_5.pkl path.")
    parser.add_argument("--sem-no-denoise", action="store_true", help="Pass --no-denoise to semantic aggregator.")
    parser.add_argument("--thickness06-px", type=int, default=2)
    parser.add_argument("--splat08-m", type=float, default=0.5)
    parser.add_argument("--splat08-block", type=int, default=1)
    # Overlap defaults (08)
    parser.add_argument("--overlap-edge-m", type=float, default=0.6)
    parser.add_argument("--overlap-prox-m", type=float, default=1.0)
    parser.add_argument("--overlap-density-k", type=int, default=2)
    parser.add_argument("--overlap-density-min", type=int, default=5)
    parser.add_argument("--overlap-close-px", type=int, default=1)
    parser.add_argument("--overlap-boost-m", type=float, default=0.4)
    parser.add_argument("--overlap08-connect-m", type=float, default=2.0)
    parser.add_argument("--overlap08-close-px", type=int, default=1)
    parser.add_argument("--overlap08-dilate-px", type=int, default=0)
    parser.add_argument("--semantic-root", default=None, help="If set, copy 08_agg_semantic.png from this root/<scene>/ to out-root/<scene>/.")
    return parser.parse_args()


def load_scene(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return mmcv.load(str(path))


def build_bank_and_traj(scene_dict: Dict) -> Tuple[Dict[str, List[np.ndarray]], List[List[np.ndarray]]]:
    vectors = {label: scene_dict.get(label, []) for label in (0, 1, 2)}
    bank = to_maptracker_bank(vectors)

    car_traj_raw = scene_dict.get("car_trajectory", [])
    car_traj: List[List[np.ndarray]] = []
    for entry in car_traj_raw:
        center = np.asarray(entry["center"], dtype=np.float32)
        yaw = float(entry["yaw_deg"])
        car_traj.append([center, yaw])
    if not car_traj:
        car_traj = [[np.array([0.0, 0.0]), 0.0]]
    return bank, car_traj


def compute_scene_bounds(banks: Sequence[Dict[str, List[np.ndarray]]]) -> Tuple[float, float, float, float]:
    combined: Dict[str, List[np.ndarray]] = {}
    for bank in banks:
        for key, value in bank.items():
            combined.setdefault(key, []).extend(value)
    return compute_bounds(combined)


def render_aggregated(
    bank: Dict[str, List[np.ndarray]],
    car_traj: List[List[np.ndarray]],
    out_path: Path,
    bounds: Tuple[float, float, float, float],
    mt_args: _Args,
) -> None:
    x_min, x_max, y_min, y_max = bounds
    plot_fig_unmerged(car_traj, x_min, x_max, y_min, y_max, str(out_path), bank, mt_args)


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.aggregated_pred)
    gt_dir = Path(args.aggregated_gt)
    static_dir = Path(args.static_gt)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.scene_list:
        scenes = [line.strip() for line in Path(args.scene_list).read_text().splitlines() if line.strip()]
    else:
        scenes = sorted(p.stem for p in pred_dir.glob("*.pkl"))

    # Use provided bounds templates only when specified.
    bounds_templates = args.bounds_pkl or []
    mt_args = _Args(args.simplify, args.line_opacity, args.dpi)

    wanted = set(args.only) if args.only else {"01", "04", "05"}

    for scene in scenes:
        pred_pkl = pred_dir / f"{scene}.pkl"
        gt_pkl = gt_dir / f"{scene}.pkl"
        static_pkl = static_dir / f"{scene}.pkl"

        # Check only what is required by requested outputs
        need_pred = "01" in wanted
        need_gt = "04" in wanted
        need_static = "05" in wanted or ("08" in wanted and args.gen_semantic)

        missing = []
        if need_pred and not pred_pkl.exists():
            missing.append("pred")
        if need_gt and not gt_pkl.exists():
            missing.append("gt")
        if need_static and not static_pkl.exists():
            missing.append("static")
        if missing:
            print(f"[skip] {scene}: missing required pickle(s) for requested panels: {','.join(missing)}")
            continue

        bank_pred = traj_pred = bank_gt = traj_gt = bank_static = traj_static = None
        if need_pred:
            pred_data = load_scene(pred_pkl)
            bank_pred, traj_pred = build_bank_and_traj(pred_data)
        if need_gt:
            gt_data = load_scene(gt_pkl)
            bank_gt, traj_gt = build_bank_and_traj(gt_data)
        if need_static:
            static_data = load_scene(static_pkl)
            bank_static, traj_static = build_bank_and_traj(static_data)

        if bounds_templates:
            tpl_paths = [str(Path(tpl.format(scene=scene))) for tpl in bounds_templates]
            bounds = compute_bounds_from_pkls(tpl_paths)
        else:
            bounds = compute_scene_bounds((bank_pred, bank_gt, bank_static))

        scene_dir = out_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)

        if "01" in wanted and bank_pred is not None:
            render_aggregated(bank_pred, traj_pred, scene_dir / "01_agg_pred_direct.png", bounds, mt_args)
        if "04" in wanted and bank_gt is not None:
            render_aggregated(bank_gt, traj_gt, scene_dir / "04_agg_gt_direct.png", bounds, mt_args)
        if "05" in wanted and bank_static is not None:
            render_static_mt_style(bank_static, scene_dir / "05_static_gt.png", *bounds, mt_args, traj_static)

        # Prefer copying 08 from a precomputed semantic root
        if "08" in wanted and args.semantic_root:
            from shutil import copyfile
            src_png = Path(args.semantic_root) / scene / "08_agg_semantic.png"
            dst_png = scene_dir / "08_agg_semantic.png"
            if src_png.exists():
                try:
                    copyfile(str(src_png), str(dst_png))
                except Exception as e:
                    print(f"[warn] copy 08 failed for {scene}: {e}")
            else:
                print(f"[warn] missing semantic 08 at {src_png}")
        # Or, optionally, generate 06/07/08 via aggregator, with 08 saved under this scene folder
        elif "08" in wanted and args.gen_semantic:
            import subprocess
            cmd = [
                "python", "maptracker/tools/tracking/aggregate_semantic_scene_sparse.py",
                args.sem_config,
                "--submission-json", args.submission_json,
                "--pos-pkl", args.pos_pkl,
                "--out-dir", str(args.out_root),
                "--scenes", scene,
                "--png", "--match-05", "--min-votes", "1", "--min-area", "8",
                "--thickness06-px", str(args.thickness06_px),
                "--overlap-ring-mode", "inner",
                "--overlap-edge-m", str(args.overlap_edge_m),
                "--overlap-prox-m", str(args.overlap_prox_m),
                "--overlap-density-k", str(args.overlap_density_k),
                "--overlap-density-min", str(args.overlap_density_min),
                "--overlap-close-px", str(args.overlap_close_px),
                "--overlap-boost-m", str(args.overlap_boost_m),
                "--overlap-08-from-06",
                "--overlap08-close-px", str(args.overlap08_close_px),
                "--overlap08-dilate-px", str(args.overlap08_dilate_px),
                "--overlap08-connect-m", str(args.overlap08_connect_m),
                "--splat08-m", str(args.splat08_m),
                "--splat08-block", str(args.splat08_block),
            ]
            if args.sem_no_denoise:
                cmd.append("--no-denoise")
            # The aggregator expects --viz-root to locate 05 when --match-05 is on
            cmd += ["--viz-root", str(args.out_root)]
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"[warn] semantic aggregation failed for {scene}: {e}")
        print(f"[ok] {scene}")


if __name__ == "__main__":
    main()
