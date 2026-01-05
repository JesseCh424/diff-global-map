import argparse
import os
import os.path as osp
import warnings

try:
    import mmcv
    from mmcv import Config, DictAction
    from mmcv.cnn import fuse_conv_bn
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
except Exception as e:  # pragma: no cover
    raise SystemExit("mmcv is required for tools/test.py: {}".format(e))

try:
    import torch
    from mmdet.apis import set_random_seed
    from mmdet.datasets import replace_ImageToTensor
    from mmdet3d.apis import single_gpu_test
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
except Exception as e:  # pragma: no cover
    raise SystemExit("mmdetection3d stack is required: {}".format(e))


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet3D test/eval a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--result-path', help='evaluate a precomputed results file (pickle)')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='fuse conv and bn (slightly faster)')
    parser.add_argument('--format-only', action='store_true', help='format results without evaluation')
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    parser.add_argument('--show', action='store_true', help='display results visually')
    parser.add_argument('--show-dir', help='directory to save visualizations')
    parser.add_argument('--gpu-collect', action='store_true', help='use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for collecting results when gpu-collect is False')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='deterministic CUDNN')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config keys')
    parser.add_argument('--options', nargs='+', action=DictAction, help='deprecated; use --eval-options')
    parser.add_argument('--eval-options', nargs='+', action=DictAction, help='kwargs for dataset.evaluate()')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.eval_options:
        raise ValueError('--options and --eval-options cannot both be specified')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def _import_plugin_from_cfg(cfg, config_path: str):
    import sys
    sys.path.append(os.path.abspath('.'))
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib

        def import_path(plugin_dir: str):
            module_dir = os.path.dirname(plugin_dir).split('/')
            module_path = module_dir[0]
            for m in module_dir[1:]:
                module_path = module_path + '.' + m
            importlib.import_module(module_path)

        if hasattr(cfg, 'plugin_dir'):
            plugin_dirs = cfg.plugin_dir
            if not isinstance(plugin_dirs, list):
                plugin_dirs = [plugin_dirs]
            for plugin_dir in plugin_dirs:
                import_path(plugin_dir)
        else:
            module_dir = os.path.dirname(config_path).split('/')
            module_path = module_dir[0]
            for m in module_dir[1:]:
                module_path = module_path + '.' + m
            importlib.import_module(module_path)


def main():
    args = parse_args()
    assert args.eval or args.format_only or args.show or args.show_dir, (
        'Specify at least one operation: --eval, --format-only, --show, or --show-dir')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    _import_plugin_from_cfg(cfg, args.config)

    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    cfg.data.test.work_dir = cfg.work_dir

    dataset = build_dataset(cfg.data.test)

    if args.result_path:
        dataset._evaluate(args.result_path)  # type: ignore[attr-defined]
        return

    # Wrapper allows us to extend behavior later without editing upstream
    try:
        from plugin.datasets.builder import build_dataloader as _build_dataloader
        data_loader = _build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data.get('shuffler_sampler', None),
            nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', None),
        )
    except Exception:
        from mmdet3d.datasets import build_dataloader as _build_dataloader
        data_loader = _build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        try:
            from plugin.core.apis.test import custom_multi_gpu_test as multi_gpu_test
        except Exception:
            from mmdet.apis import multi_gpu_test  # fallback
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            if args.eval_options is not None:
                eval_kwargs.update(args.eval_options)
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            print('start evaluation!')
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()

