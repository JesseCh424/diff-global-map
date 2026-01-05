from global_diffusion_map.plugin.configs.global_diffusion.av2_polydiffuse_official_base import *  # noqa

# Override dataset roots to point to the small overfit test subset.
# - static vectors (simplified): maptracker/work_dirs/test/av2_old/train/gt_simp
# - conditioning rasters (11):   maptracker/work_dirs/test/av2_old/train/condition

STATIC_ROOT = 'maptracker/work_dirs/test/av2_old/train/gt_simp'
RENDERED_GT_ROOT = 'maptracker/work_dirs/test/av2_old/train/condition'
# Align with official PolyDiffuse: train on GT only; no mixed-init.
INIT_ROOT = None
# No semantic needed for overfit; keep but unused when use_condition='11'
SEMANTIC_ROOT = None
# Limit to a single scene list for overfit
SCENE_LIST = 'global_diffusion_map/work_dirs/overfit_one_scene.txt'

# Rebuild train data dict with 11-only conditioning and zero mix prob
data["train"] = dict(
    type='AV2GlobalDiffusionDataset',
    static_root=STATIC_ROOT,
    rendered_gt_root=RENDERED_GT_ROOT,
    init_root=INIT_ROOT,
    semantic_root=SEMANTIC_ROOT,
    scene_list=SCENE_LIST,
    use_condition='11',
    mix_with_08_prob=0.0,
    M=M,
    num_queries=NUM_QUERIES,
    class_budget=CLASS_BUDGET,
    # Overfitting single scene: disable random instance drop for stability
    drop_instance=False,
    load_image=True,
    cond_max_side=1024,
    cond_fixed_size=(1024, 1024),
    pad_fill='gaussian',
    pad_sigma=0.2,
    seed=1234,
)

# Keep validation configuration as-is; in-loop validation is disabled by default
# unless VAL_EVERY_TICKS env var is set.
