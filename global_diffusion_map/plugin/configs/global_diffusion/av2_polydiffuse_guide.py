from .av2_polydiffuse_base import *  # noqa

# Guidance stage uses vectors only; disable image loading and instance drop.
data['train'] = make_data('train', mode='guide')
data['val'] = make_data('val', mode='guide')
data['test'] = make_data('val', mode='guide')

runner = dict(type='EpochBasedRunner', max_epochs=1)
work_dir = './work_dirs/av2_polydiffuse_guide'

