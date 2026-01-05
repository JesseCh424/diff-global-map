# Default runtime and plugin wiring (scaffold)

custom_imports = dict(
    imports=['global_diffusion_map.plugin'],
    allow_failed_imports=True,
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
cudnn_benchmark = True

# Ensure local plugin import without editing entrypoints
plugin = True
plugin_dir = 'global_diffusion_map/plugin'

