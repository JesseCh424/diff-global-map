from global_diffusion_map.plugin.configs.global_diffusion.av2_polydiffuse_official_base import *  # noqa

# Restrict to one-scene list for overfitting guidance
SCENE_LIST = 'global_diffusion_map/work_dirs/overfit_one_scene.txt'

# Rebuild datasets for guide mode with the updated SCENE_LIST
data['train'] = make_data('guide')
data['val'] = make_data('guide')
data['test'] = make_data('guide')

runner = dict(type='EpochBasedRunner', max_epochs=1)
work_dir = './work_dirs/av2_polydiffuse_official_guide_overfit'

