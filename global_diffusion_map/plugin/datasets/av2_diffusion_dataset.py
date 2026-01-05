"""AV2 Global Diffusion Dataset

Produces per-scene vector targets and raster conditions for diffusion training.

Returned fields per sample (to match PolyDiffuse loops):
- gt_bboxes_3d: float32 [num_queries, M, 2] in [-1, 1]
- gt_labels_3d: int64   [num_queries]
- pts_mask: bool        [num_queries, M]  True = padding (ignored)
- img: float32 [len_queue=1, num_cams=1, 3, H, W] if load_image=True else zeros
- img_metas: list of length 1 (queue) per sample; kept but unused by our model

Notes
- Uses bounds saved in static GT pickles under key "bounds" = [minx, miny, maxx, maxy].
- Vectors are uniformly resampled to M points.
- Scene list can be passed as a text file with one scene id per line, or left None to scan roots.
"""
from __future__ import annotations

import os
import os.path as osp
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

from shapely.geometry import LineString

from mmdet.datasets import DATASETS


def _load_pickle(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _uniform_sample(line: LineString, num: int) -> np.ndarray:
    if line.length <= 1e-6:
        # degenerate: repeat the point
        p = np.array(line.coords[0], dtype=np.float32)
        return np.tile(p[None, :], (num, 1))
    dists = np.linspace(0.0, line.length, num=num, dtype=np.float32)
    pts = [list(line.interpolate(float(d)).coords)[0] for d in dists]
    return np.asarray(pts, dtype=np.float32)


def _normalize_xy(xy: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)
    out = np.empty_like(xy, dtype=np.float32)
    out[:, 0] = ((xy[:, 0] - minx) / w) * 2.0 - 1.0
    out[:, 1] = ((xy[:, 1] - miny) / h) * 2.0 - 1.0
    return out


def _pad_instances(inst_list: List[np.ndarray], label_list: List[int], M: int, num_queries: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Pack into fixed [num_queries, M, 2] + masks
    num = min(len(inst_list), num_queries)
    pts = np.zeros((num_queries, M, 2), dtype=np.float32)
    mask = np.ones((num_queries, M), dtype=bool)  # True = padding
    labels = np.zeros((num_queries,), dtype=np.int64)

    if num > 0:
        for i in range(num):
            cur = inst_list[i]
            pts[i, :, :] = cur
            mask[i, :] = False
            labels[i] = int(label_list[i])
    return pts, labels, mask


def _read_scene_list(scene_list: Optional[str], root_glob: Optional[str]) -> List[str]:
    if scene_list and osp.exists(scene_list):
        with open(scene_list, 'r') as f:
            return [ln.strip() for ln in f if ln.strip()]
    if root_glob and osp.isdir(root_glob):
        return sorted([osp.splitext(x)[0] for x in os.listdir(root_glob) if x.endswith('.pkl')])
    return []


@DATASETS.register_module()
class AV2GlobalDiffusionDataset:  # minimal Dataset API for mmdet3d build_dataset
    CLASSES = ['ped_crossing', 'divider', 'boundary']

    def __init__(
        self,
        static_root: str,
        rendered_gt_root: Optional[str] = None,
        semantic_root: Optional[str] = None,
        init_root: Optional[str] = None,
        scene_list: Optional[str] = None,
        use_condition: str = '11',  # '11' | '08' | 'mix'
        mix_with_08_prob: float = 0.0,
        M: int = 32,
        num_queries: int = 512,
        drop_instance: bool = False,
        load_image: bool = True,
        seed: int = 0,
        class_budget: Optional[Dict[int, int]] = None,
        pad_fill: str = 'zero',
        pad_sigma: float = 0.2,
        cond_max_side: Optional[int] = None,
        cond_fixed_size: Optional[Tuple[int, int]] = None,
        proposal_root: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.static_root = static_root
        self.rendered_gt_root = rendered_gt_root
        self.semantic_root = semantic_root
        self.init_root = init_root
        self.scene_ids = _read_scene_list(scene_list, static_root)
        self.use_condition = use_condition
        self.mix_with_08_prob = float(mix_with_08_prob)
        import os as _os
        _M_override = _os.environ.get('AV2_M_OVERRIDE')
        _N_override = _os.environ.get('AV2_NUM_QUERIES_OVERRIDE')
        self.M = int(_M_override) if _M_override else int(M)
        self.num_queries = int(_N_override) if _N_override else int(num_queries)
        self.drop_instance = bool(drop_instance)
        self.load_image = bool(load_image)
        self.rng = np.random.RandomState(seed)
        self.pad_fill = str(pad_fill).lower().strip()
        self.pad_sigma = float(pad_sigma)
        self.cond_max_side = int(cond_max_side) if cond_max_side else None
        self.cond_fixed_size = tuple(cond_fixed_size) if cond_fixed_size else None
        self.proposal_root = proposal_root
        # default per-class budget if provided
        self.class_budget = class_budget or {0: 80, 1: 176, 2: 256}
        # Scale budgets if sum exceeds num_queries
        total = sum(self.class_budget.values())
        if total > self.num_queries and total > 0:
            scale = self.num_queries / float(total)
            new = {k: max(1, int(round(v * scale))) for k, v in self.class_budget.items()}
            # adjust to exact sum
            diff = self.num_queries - sum(new.values())
            keys = list(new.keys())
            for i in range(abs(diff)):
                idx = keys[i % len(keys)]
                new[idx] += 1 if diff > 0 else -1
            self.class_budget = new

        # Enable lightweight per-scene caching for guidance mode
        # Only when we are not dropping instances and not loading images
        self._enable_cache: bool = (not self.drop_instance) and (not self.load_image)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.scene_ids)

    def _choose_condition_path(self, scene: str) -> Optional[str]:
        # Decide 11 vs 10 vs 08
        if not self.load_image:
            return None
        choice = self.use_condition
        if self.use_condition == 'mix' and self.rng.rand() < self.mix_with_08_prob:
            choice = '08'
        if choice == '11':
            if not self.rendered_gt_root:
                return None
            p = osp.join(self.rendered_gt_root, scene, '11_gt_aug.png')
            return p
        elif choice == '10':
            if not self.rendered_gt_root:
                return None
            p = osp.join(self.rendered_gt_root, scene, '10_render_gt.png')
            return p
        elif choice == '08':
            if not self.semantic_root:
                return None
            p = osp.join(self.semantic_root, scene, '08_agg_semantic.png')
            return p
        return None

    def _load_raster(self, path: str) -> np.ndarray:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        # Optional downscale to limit max side first
        if self.cond_max_side is not None:
            ms = max(w, h)
            if ms > self.cond_max_side and ms > 0:
                scale = self.cond_max_side / float(ms)
                w = max(1, int(round(w * scale)))
                h = max(1, int(round(h * scale)))
                try:
                    img = img.resize((w, h), Image.Resampling.LANCZOS)
                except Exception:
                    img = img.resize((w, h))
        # Optional letterbox to fixed canvas to guarantee batching
        if self.cond_fixed_size is not None:
            tgt_h, tgt_w = self.cond_fixed_size
            # scale to fit inside target while preserving aspect
            if w == 0 or h == 0:
                canvas = Image.new('RGB', (tgt_w, tgt_h), (0, 0, 0))
                img = canvas
            else:
                scale = min(tgt_w / float(w), tgt_h / float(h))
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                if (new_w, new_h) != (w, h):
                    try:
                        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    except Exception:
                        img_resized = img.resize((new_w, new_h))
                else:
                    img_resized = img
                canvas = Image.new('RGB', (tgt_w, tgt_h), (0, 0, 0))
                # paste top-left (or center if preferred). Use top-left for determinism.
                canvas.paste(img_resized, (0, 0))
                img = canvas
        arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,3 in [0,1]
        # Return as (1,1,3,H,W)
        arr = arr.transpose(2, 0, 1)  # C,H,W
        return arr[None, None, ...]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene = self.scene_ids[idx]
        static_pkl = osp.join(self.static_root, f'{scene}.pkl')
        if not osp.exists(static_pkl):
            raise FileNotFoundError(static_pkl)
        data = _load_pickle(static_pkl)

        # Bounds
        if 'bounds' in data:
            bounds = data['bounds']
        else:
            # Fallback: derive from all points
            pts_all = []
            for k in (0, 1, 2):
                for arr in data.get(k, []):
                    pts_all.append(arr)
            if len(pts_all) == 0:
                bounds = [0, 0, 1, 1]
            else:
                cat = np.concatenate(pts_all, axis=0)
                minx, miny = cat.min(0)
                maxx, maxy = cat.max(0)
                bounds = [float(minx), float(miny), float(maxx), float(maxy)]

        # Build or reuse per-scene packed tensors for targets (GT)
        if self._enable_cache and scene in self._cache:
            pts, labels, mask = self._cache[scene]
        else:
            inst_list: List[np.ndarray] = []
            label_list: List[int] = []
            for cls_id in (2, 1, 0):  # boundary, divider, ped_crossing
                vecs = data.get(cls_id, [])
                scored: List[Tuple[float, np.ndarray]] = []
                for v in vecs:
                    try:
                        ln = LineString(v).length
                    except Exception:
                        ln = 0.0
                    scored.append((ln, v))
                scored.sort(key=lambda x: x[0], reverse=True)
                cap = self.class_budget.get(cls_id, 0)
                take = scored[:cap]
                for _, pts_arr in take:
                    line = LineString(pts_arr)
                    sampled = _uniform_sample(line, self.M)
                    normed = _normalize_xy(sampled, bounds)
                    inst_list.append(normed)
                    # Map to MapTR class order: ['divider', 'ped_crossing', 'boundary'] -> 0/1/2
                    if cls_id == 1:
                        mapped = 0
                    elif cls_id == 0:
                        mapped = 1
                    else:
                        mapped = 2
                    label_list.append(mapped)

            pts, labels, mask = _pad_instances(inst_list, label_list, self.M, self.num_queries)
            # Optional Gaussian-noise padding for padded queries
            if self.pad_fill == 'gaussian':
                num_valid = min(len(inst_list), self.num_queries)
                if num_valid < self.num_queries:
                    noise = np.random.randn(self.num_queries - num_valid, self.M, 2).astype(np.float32)
                    noise *= self.pad_sigma
                    # Coordinates are normalized to [-1, 1]
                    noise = np.clip(noise, -1.0, 1.0)
                    pts[num_valid:, :, :] = noise
            if self._enable_cache:
                # store numpy arrays to minimize memory, torch conversion happens later
                self._cache[scene] = (pts, labels, mask)

        # Optional: build an initial state z_init from a separate root (e.g., aggregated preds)
        init_pts_np: Optional[np.ndarray] = None
        if self.init_root:
            init_pkl = osp.join(self.init_root, f'{scene}.pkl')
            if osp.exists(init_pkl):
                try:
                    init_data = _load_pickle(init_pkl)
                    inst_list_i: List[np.ndarray] = []
                    label_list_i: List[int] = []
                    for cls_id in (2, 1, 0):  # boundary, divider, ped -> map to MapTR order later
                        vecs = init_data.get(cls_id, [])
                        scored: List[Tuple[float, np.ndarray]] = []
                        for v in vecs:
                            try:
                                ln = LineString(v).length
                            except Exception:
                                ln = 0.0
                            scored.append((ln, v))
                        scored.sort(key=lambda x: x[0], reverse=True)
                        cap = self.class_budget.get(cls_id, 0)
                        take = scored[:cap]
                        for _, pts_arr in take:
                            line = LineString(pts_arr)
                            sampled = _uniform_sample(line, self.M)
                            normed = _normalize_xy(sampled, bounds)
                            inst_list_i.append(normed)
                            # Map to MapTR class order: divider=0, ped=1, boundary=2
                            if cls_id == 1:
                                mapped = 0
                            elif cls_id == 0:
                                mapped = 1
                            else:
                                mapped = 2
                            label_list_i.append(mapped)
                    init_pts_np, _labels_i, _mask_i = _pad_instances(inst_list_i, label_list_i, self.M, self.num_queries)
                except Exception:
                    init_pts_np = None

        # Condition raster
        if self.load_image:
            cond_path = self._choose_condition_path(scene)
            if cond_path and osp.exists(cond_path):
                img = self._load_raster(cond_path)
            else:
                # dummy image if missing; respect fixed canvas if configured
                if self.cond_fixed_size is not None:
                    tgt_h, tgt_w = self.cond_fixed_size
                    img = np.zeros((1, 1, 3, int(tgt_h), int(tgt_w)), dtype=np.float32)
                else:
                    side = int(self.cond_max_side) if self.cond_max_side else 64
                    img = np.zeros((1, 1, 3, side, side), dtype=np.float32)
        else:
            if self.cond_fixed_size is not None:
                tgt_h, tgt_w = self.cond_fixed_size
                img = np.zeros((1, 1, 3, int(tgt_h), int(tgt_w)), dtype=np.float32)
            else:
                side = int(self.cond_max_side) if self.cond_max_side else 64
                img = np.zeros((1, 1, 3, side, side), dtype=np.float32)

        # img_metas: list[dict] (per sample queue); collate -> list[list[dict]] then MapTR picks the last queue entry
        H = int(img.shape[-2])
        W = int(img.shape[-1])
        # Provide identity lidar2img and zero can_bus; MapTR expects list[dict]
        img_metas = [{
            'scene_id': scene,
            # MapTR expects a per-camera list of (H,W); we use one camera.
            'img_shape': [(H, W)],
            'pc_range': [-1.0, -1.0, -2.0, 1.0, 1.0, 2.0],  # normalized
            'bounds': bounds,
            'lidar2img': np.eye(4, dtype=np.float32)[None, ...],  # (num_cams=1,4,4)
            'can_bus': np.zeros(18, dtype=np.float32),
        }]

        # Convert to torch.Tensor for safer collation under multi-worker loaders
        item = {
            'gt_bboxes_3d': torch.from_numpy(pts.astype(np.float32)),
            'gt_labels_3d': torch.from_numpy(labels.astype(np.int64)),
            'pts_mask': torch.from_numpy(mask.astype(np.bool_)),
            'img': torch.from_numpy(img.astype(np.float32)),
            'img_metas': img_metas,
            'scene_id': scene,
        }
        # Load packed proposals (if available)
        if self.proposal_root:
            pp = osp.join(self.proposal_root, f'{scene}.npz')
            if osp.exists(pp):
                try:
                    arr = np.load(pp)
                    item['proposal_pts'] = torch.from_numpy(arr['pts'].astype(np.float32))
                    item['proposal_mask'] = torch.from_numpy(arr['mask'].astype(np.bool_))
                    item['proposal_labels'] = torch.from_numpy(arr['labels'].astype(np.int64))
                    item['proposal_bounds'] = torch.from_numpy(arr['bounds'].astype(np.float32))
                except Exception:
                    pass
        if init_pts_np is not None:
            item['init_bboxes_3d'] = torch.from_numpy(init_pts_np.astype(np.float32))
        return item
