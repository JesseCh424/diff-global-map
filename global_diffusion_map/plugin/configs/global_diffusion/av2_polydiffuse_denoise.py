from .av2_polydiffuse_base import *  # noqa

# Denoising stage uses raster conditions; enable image loading and optional instance drop.
data['train'] = make_data('train', mode='denoise')
data['val'] = make_data('val', mode='denoise')
data['test'] = make_data('val', mode='denoise')

runner = dict(type='EpochBasedRunner', max_epochs=1)
work_dir = './work_dirs/av2_polydiffuse_denoise'

