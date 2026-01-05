from global_diffusion_map.plugin.configs.global_diffusion.av2_polydiffuse_official_base import *  # noqa

# Override roots to use unsimplified GT and condition=10_render_gt.png only.
STATIC_ROOT = 'maptracker/work_dirs/static_gt_vector/av2_oldsplit/train'
RENDERED_GT_ROOT = 'maptracker/work_dirs/rendered_gt/av2_oldsplit/train'

# Rebuild train dataset dict to force use_condition='10'.
data["train"] = dict(
    type='AV2GlobalDiffusionDataset',
    static_root=STATIC_ROOT,
    rendered_gt_root=RENDERED_GT_ROOT,
    semantic_root=SEMANTIC_ROOT,
    scene_list=SCENE_LIST,
    use_condition='10',
    mix_with_08_prob=0.0,
    M=M,
    num_queries=NUM_QUERIES,
    class_budget=CLASS_BUDGET,
    drop_instance=True,
    load_image=True,
    cond_max_side=1024,
    cond_fixed_size=(1024, 1024),
    pad_fill='gaussian',
    pad_sigma=0.2,
    seed=1234,
)

