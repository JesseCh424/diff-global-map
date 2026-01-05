import os
import json

from global_diffusion_map.plugin.datasets.av2_diffusion_dataset import AV2GlobalDiffusionDataset  # noqa: F401

_base_ = ['av2_polydiffuse_official_base.py']

# Overwrite roots for overfit-10 policy (unsimplified GT + 10 condition)

TRAIN_STATIC = 'maptracker/work_dirs/static_gt_vector/av2_oldsplit/train'
VAL_STATIC   = 'maptracker/work_dirs/static_gt_vector/av2_oldsplit/val'

TRAIN_RENDER = 'maptracker/work_dirs/rendered_gt/av2_oldsplit/train'
VAL_RENDER   = 'maptracker/work_dirs/rendered_gt/av2_oldsplit/val'

TRAIN_LIST = 'global_diffusion_map/work_dirs/overfit/train_one.txt'
VAL_LIST   = 'global_diffusion_map/work_dirs/overfit/val_one.txt'

PACKED_TRAIN = 'global_diffusion_map/work_dirs/packed_proposals/av2_oldsplit/overfit_train'

# Load caps
STATS_JSON = os.environ.get('AV2_STATS_JSON', 'global_diffusion_map/work_dirs/av2_stats.json')
if os.path.exists(STATS_JSON):
    with open(STATS_JSON, 'r') as f:
        _stats = json.load(f)
    try:
        del f
    except Exception:
        pass
    M = int(_stats.get('M', 30))
    NUM_QUERIES = int(_stats.get('num_queries', 64))
    CLASS_BUDGET = {int(k): int(v) for k, v in _stats.get('class_budget', {0: 8, 1: 17, 2: 25}).items()}
else:
    M = 30
    NUM_QUERIES = 64
    CLASS_BUDGET = {0: 8, 1: 17, 2: 25}

# Override dataset sections from base
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='AV2GlobalDiffusionDataset',
        static_root=TRAIN_STATIC,
        rendered_gt_root=TRAIN_RENDER,
        semantic_root=None,
        scene_list=TRAIN_LIST,
        use_condition='10',
        M=M,
        num_queries=NUM_QUERIES,
        class_budget=CLASS_BUDGET,
        drop_instance=True,
        load_image=True,
        cond_max_side=1024,
        cond_fixed_size=(1024, 1024),
        # Revert to official: do not use proposals in training by default
        proposal_root=None,
        pad_fill='gaussian', pad_sigma=0.2,
        seed=1234,
    ),
    val=dict(
        type='AV2GlobalDiffusionDataset',
        static_root=VAL_STATIC,
        rendered_gt_root=VAL_RENDER,
        semantic_root=None,
        scene_list=VAL_LIST,
        use_condition='10',
        M=M,
        num_queries=NUM_QUERIES,
        class_budget=CLASS_BUDGET,
        drop_instance=False,
        load_image=True,
        cond_max_side=1024,
        cond_fixed_size=(1024, 1024),
        proposal_root=None,
        pad_fill='gaussian', pad_sigma=0.2,
        seed=1234,
    ),
    test=dict(
        type='AV2GlobalDiffusionDataset',
        static_root=VAL_STATIC,
        rendered_gt_root=VAL_RENDER,
        semantic_root=None,
        scene_list=VAL_LIST,
        use_condition='10',
        M=M,
        num_queries=NUM_QUERIES,
        class_budget=CLASS_BUDGET,
        drop_instance=False,
        load_image=True,
        cond_max_side=1024,
        cond_fixed_size=(1024, 1024),
        pad_fill='gaussian', pad_sigma=0.2,
        seed=1234,
    ),
)

work_dir = './work_dirs/av2_polydiffuse_overfit_10'
