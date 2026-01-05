from global_diffusion_map.plugin.configs.global_diffusion.av2_polydiffuse_official_base import *  # noqa

# Use full static GT vectors (unsimplified) for guide training on train split
STATIC_ROOT = 'maptracker/work_dirs/static_gt_vector/av2_oldsplit/train'
SCENE_LIST = None  # full dataset

# Rebuild datasets for guide mode with updated STATIC_ROOT
data['train'] = make_data('guide')
data['val'] = make_data('guide')
data['test'] = make_data('guide')

runner = dict(type='EpochBasedRunner', max_epochs=1)
work_dir = './work_dirs/av2_polydiffuse_official_guide_full_rawgt'

