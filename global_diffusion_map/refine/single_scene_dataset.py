from __future__ import annotations

import os
import os.path as osp
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset_refine import RefineCaps, _normalize_xy, _uniform_resample, pack_gt_to_slots


def load_pickle(path: str) -> Dict[str, Any]:
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_bounds_from_any(d: Dict[str, Any]) -> List[float]:
    """Strict bounds fetch: only accept serialized bounds; no fallback."""
    if 'bounds' in d and d['bounds'] is not None:
        b = d['bounds']
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    raise RuntimeError('bounds-missing: expected canonical bounds in pickle')


def load_raster_png(path: str, out_size: Tuple[int, int] | None = None) -> np.ndarray:
    """Load raster without pre-resize; keep original size.
    Letterbox will be applied downstream for visualization or the encoder
    will consume variable sizes (AdaptiveAvgPool)."""
    img = Image.open(path).convert('RGB')
    if out_size is not None:
        img = img.resize((out_size[1], out_size[0]), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # [3,H,W]


class SingleSceneDataset(Dataset):
    """Single-scene overfit dataset with on-the-fly proposal simulation.

    Each __getitem__ returns a different noisy proposal while keeping
    the same raster and GT target pack.
    """

    def __init__(
        self,
        static_root: str,
        rendered_root: str,
        agg_pred_root: str,
        scene: str,
        caps: RefineCaps,
        class_budgets: Dict[int, int],
        length: int = 1000,
        jitter_sigma_m: float = 0.5,
        drop_rate: float = 0.0,
        ghosts: int = 0,
    ) -> None:
        super().__init__()
        self.length = int(length)
        self.caps = caps
        self.jitter_sigma_m = float(jitter_sigma_m)
        self.drop_rate = float(drop_rate)
        self.ghosts = int(ghosts)

        gt_pkl = osp.join(static_root, f'{scene}.pkl')
        gt = load_pickle(gt_pkl)
        bounds = gt.get('bounds')
        if bounds is None:
            raise RuntimeError(f"bounds-missing: canonical static GT bounds not found for scene={scene}")
        self.bounds = bounds
        self.gt = gt
        self.budgets = {int(k): int(v) for k, v in class_budgets.items()}
        self.gt_pack, self.gt_mask, self.gt_present = pack_gt_to_slots(
            gt, bounds, self.budgets, num_points=caps.num_points, num_queries=caps.num_queries)

        raster_path = osp.join(rendered_root, scene, '10_render_gt.png')
        self.raster = load_raster_png(raster_path)  # [3,H,W]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Strong jitter proposal around GT; optionally drop or add ghosts
        prop = self.gt_pack.copy()
        noise = np.random.normal(scale=self.jitter_sigma_m, size=prop.shape).astype(np.float32)
        prop = np.clip(prop + noise, -1.0, 1.0)
        # Pack to tensors
        x = torch.from_numpy(prop).float()                 # [N,P,2]
        r = torch.from_numpy(self.raster).float()          # [3,H,W]
        tgt_c = torch.from_numpy(self.gt_pack).float()     # [N,P,2]
        tgt_m = torch.from_numpy(self.gt_mask).bool()      # [N,P]
        tgt_p = torch.from_numpy(self.gt_present).long()   # [N]
        return {
            'proposal': x,
            'raster': r,
            'tgt_coords': tgt_c,
            'tgt_mask': tgt_m,
            'tgt_present': tgt_p,
        }


class DynamicOverfitDataset(Dataset):
    """Super overfitting dataset with dynamic modes per iteration.

    Modes:
      - delete_test: add random ghost curves on top of jittered GT (expect deletion)
      - create_test: drop some GT instances, fill with noise (expect creation)
      - refine_test: jitter all GT instances (expect refinement)
    """

    def __init__(
        self,
        static_root: str,
        rendered_root: str,
        agg_pred_root: str,
        scene: str,
        caps: RefineCaps,
        class_budgets: Dict[int, int],
        length: int = 2000,
        jitter_sigma_m: float = 0.6,
        drop_frac_range: Sequence[float] = (0.3, 0.5),
        ghosts_range: Sequence[int] = (3, 6),
    ) -> None:
        super().__init__()
        self.length = int(length)
        self.caps = caps
        self.jitter_sigma_m = float(jitter_sigma_m)
        self.drop_lo, self.drop_hi = float(drop_frac_range[0]), float(drop_frac_range[1])
        self.gh_lo, self.gh_hi = int(ghosts_range[0]), int(ghosts_range[1])

        gt_pkl = osp.join(static_root, f'{scene}.pkl')
        gt = load_pickle(gt_pkl)
        bounds = gt.get('bounds')
        if bounds is None:
            raise RuntimeError(f"bounds-missing: canonical static GT bounds not found for scene={scene}")
        self.bounds = bounds
        self.budgets = {int(k): int(v) for k, v in class_budgets.items()}
        # pack GT once
        self.gt_pack, self.gt_mask, self.gt_present = pack_gt_to_slots(
            gt, bounds, self.budgets, num_points=caps.num_points, num_queries=caps.num_queries)
        # raster
        raster_path = osp.join(rendered_root, scene, '10_render_gt.png')
        self.raster = load_raster_png(raster_path)

    def __len__(self) -> int:
        return self.length

    def _rand_ghost(self, P: int) -> np.ndarray:
        # random polyline in [-1,1]^2 with mild continuity
        pts = np.random.uniform(-1.0, 1.0, size=(P, 2)).astype(np.float32)
        # smooth a bit
        for k in range(1, P):
            pts[k] = 0.7 * pts[k] + 0.3 * pts[k - 1]
        return pts

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import random
        N, P = self.caps.num_queries, self.caps.num_points
        mode = random.choice(['delete_test', 'create_test', 'refine_test'])
        # start from GT pack
        prop = self.gt_pack.copy()
        present = ~self.gt_mask.all(axis=1)

        if mode == 'delete_test':
            # jitter GT a bit so matching isn't trivial
            prop[present] = np.clip(prop[present] + np.random.normal(scale=self.jitter_sigma_m, size=prop[present].shape).astype(np.float32), -1.0, 1.0)
            # add ghosts into empty slots
            k = np.random.randint(self.gh_lo, self.gh_hi + 1)
            empty = np.where(~present)[0].tolist()
            random.shuffle(empty)
            for i in empty[:k]:
                prop[i] = self._rand_ghost(P)
        elif mode == 'create_test':
            # drop some GT instances
            ids = np.where(present)[0].tolist()
            random.shuffle(ids)
            drop_num = max(1, int(round(len(ids) * np.random.uniform(self.drop_lo, self.drop_hi))))
            for i in ids[:drop_num]:
                prop[i] = np.random.normal(size=(P, 2)).astype(np.float32)  # noise slot
        else:  # refine_test
            prop[present] = np.clip(prop[present] + np.random.normal(scale=self.jitter_sigma_m, size=prop[present].shape).astype(np.float32), -1.0, 1.0)

        # tensors
        x = torch.from_numpy(prop).float()
        r = torch.from_numpy(self.raster).float()
        tgt_c = torch.from_numpy(self.gt_pack).float()
        tgt_m = torch.from_numpy(self.gt_mask).bool()
        tgt_p = torch.from_numpy(self.gt_present).long()
        return {
            'proposal': x,
            'raster': r,
            'tgt_coords': tgt_c,
            'tgt_mask': tgt_m,
            'tgt_present': tgt_p,
        }
