# ---------------------------------------------
# Scaffolded training wrapper (MMDetection3D-style)
# ---------------------------------------------
from __future__ import annotations

import argparse
import copy
import os
import time
import warnings
from os import path as osp

try:
    import mmcv
    from mmcv import Config, DictAction
    from mmcv.runner import get_dist_info, init_dist
except Exception as e:  # pragma: no cover
    raise SystemExit("mmcv is required for tools/train.py: {}".format(e))

try:
    import torch
    from mmdet import __version__ as mmdet_version
    from mmdet.apis import set_random_seed
    from mmseg import __version__ as mmseg_version
    from mmdet3d import __version__ as mmdet3d_version
    from mmdet3d.apis import train_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet3d.utils import collect_env, get_root_logger
    from mmcv.utils import TORCH_VERSION, digit_version
except Exception as e:  # pragma: no cover
    raise SystemExit("mmdetection3d stack is required: {}".format(e))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true', help='disable validation during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int, help='number of gpus (non-distributed)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus (non-distributed)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='deterministic CUDNN')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config keys')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true', help='scale lr with number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
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
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    _import_plugin_from_cfg(cfg, args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer.get('type') == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2'  # known fix upstream
    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger_name = 'mmseg' if cfg.model.get('type') == 'EncoderDecoder3D' else 'mmdet'
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, name=logger_name)

    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    cfg.data.train.work_dir = cfg.work_dir
    cfg.data.val.work_dir = cfg.work_dir
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=None,
            PALETTE=getattr(datasets[0], 'PALETTE', None),
        )

    # use plugin thin wrapper (for future customization)
    try:
        from plugin.core.apis import custom_train_model
        train_entry = custom_train_model
    except Exception:
        train_entry = train_model

    train_entry(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == '__main__':
    main()

