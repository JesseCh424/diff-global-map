import os
import json

from global_diffusion_map.plugin.datasets.av2_diffusion_dataset import AV2GlobalDiffusionDataset  # noqa: F401

_base_ = ['../_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# Data roots (AV2)
# Default to unsimplified static GT for supervision
STATIC_ROOT = 'maptracker/work_dirs/static_gt_vector/av2_oldsplit/train'
RENDERED_GT_ROOT = 'maptracker/work_dirs/rendered_gt/av2_oldsplit/train'
SEMANTIC_ROOT = 'maptracker/work_dirs/semantic/train'
SCENE_LIST = None

# Validation roots (true val split)
VAL_STATIC_ROOT = 'maptracker/work_dirs/static_gt_vector/av2_oldsplit/val'
# For validation, 08 is also available under rendered_gt val root; point both to rendered_gt val
VAL_RENDERED_GT_ROOT = 'maptracker/work_dirs/rendered_gt/av2_oldsplit/val'
VAL_SEMANTIC_ROOT = 'maptracker/work_dirs/rendered_gt/av2_oldsplit/val'

# Load stats JSON (produced by tools/scan_av2_stats.py)
STATS_JSON = os.environ.get('AV2_STATS_JSON', 'global_diffusion_map/work_dirs/av2_stats.json')
if os.path.exists(STATS_JSON):
    with open(STATS_JSON, 'r') as f:
        _stats = json.load(f)
    try:
        del f
    except Exception:
        pass
    M = int(_stats.get('M', 32))
    NUM_QUERIES = int(_stats.get('num_queries', 512))
    CLASS_BUDGET = {int(k): int(v) for k, v in _stats.get('class_budget', {0: 80, 1: 176, 2: 256}).items()}
else:
    # PolyDiffuse defaults when stats JSON is absent
    M = 20
    NUM_QUERIES = 50
    # Fallback class budget approximated to sum to 50 (ped, divider, boundary)
    CLASS_BUDGET = {0: 8, 1: 17, 2: 25}


point_cloud_range = [-1.0, -1.0, -2.0, 1.0, 1.0, 2.0]

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
# BEV resolution (wide scenes: allocate more tokens in X)
bev_h_ = 100
bev_w_ = 200

map_classes = ['divider', 'ped_crossing', 'boundary']
num_map_classes = len(map_classes)

model = dict(
    type='MapTR',
    use_grid_mask=True,
    video_test_mode=False,
    # Official-aligned backbone (ResNet-50) with FPN in_channels=2048
    img_backbone=dict(type='ResNet', depth=50, num_stages=4, out_indices=(3,), frozen_stages=1,
                      norm_cfg=dict(type='BN', requires_grad=False), norm_eval=True, style='pytorch'),
    img_neck=dict(type='FPN', in_channels=[2048], out_channels=_dim_, start_level=0,
                  add_extra_convs='on_output', num_outs=_num_levels_, relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='MapTRHead', bev_h=bev_h_, bev_w=bev_w_, num_vec=NUM_QUERIES, num_pts_per_vec=M, num_pts_per_gt_vec=M,
        dir_interval=1, query_embed_type='instance_pts', transform_method='minmax', gt_shift_pts_pattern='v2',
        num_classes=num_map_classes, in_channels=_dim_, sync_cls_avg_factor=True, with_box_refine=True,
        as_two_stage=False, code_size=2, code_weights=[1.0, 1.0, 1.0, 1.0],
        # Revert to official: disable ProposalEncoder fusion by default
        use_proposal=False,
        transformer=dict(
            type='MapTRPerceptionTransformer', num_cams=1, rotate_prev_bev=True, use_shift=True, use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder', num_layers=1, pc_range=point_cloud_range, num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(type='TemporalSelfAttention', embed_dims=_dim_, num_levels=1),
                        dict(
                            type='GeometrySptialCrossAttention', pc_range=point_cloud_range, num_cams=1,
                            attention=dict(type='GeometryKernelAttention', embed_dims=_dim_, num_heads=4,
                                           dilation=1, kernel_size=(3, 5), num_levels=_num_levels_),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_, ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='MapTRDecoder', num_layers=6,  # official depth
                return_intermediate=True,
                timestep_embed=128 * 4,
                transformerlayers=dict(
                    type='PolyDetrTransformerDecoderLayer',
                    # Ensure per-poly decode uses the same M as dataset caps
                    num_verts=M,
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1),
                        dict(type='MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1),
                        dict(type='CustomMSDeformableAttention', embed_dims=_dim_, num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_, ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        bbox_coder=dict(type='MapTRNMSFreeCoder', post_center_range=[-2, -2, -2, 2, 2, 2, 2, 2], pc_range=point_cloud_range,
                        max_num=NUM_QUERIES, voxel_size=[0.1, 0.1, 4], num_classes=num_map_classes),
        positional_encoding=dict(type='LearnedPositionalEncoding', num_feats=_pos_dim_, row_num_embed=bev_h_, col_num_embed=bev_w_),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss', loss_weight=5.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1], voxel_size=[0.15, 0.15, 4], point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(type='MapTRAssigner', cls_cost=dict(type='FocalLossCost', weight=2.0),
                      reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                      iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                      pts_cost=dict(type='OrderedPtsL1Cost', weight=5), pc_range=point_cloud_range))))


def make_data(mode: str):
    # Default conditioning raster: 10_render_gt.png (unsimplified GT render)
    _use_condition = '10'
    _mix_prob = 0.0
    # Revert to official: do not use proposal_root by default in training/val/test
    _proposal_root = None
    return dict(
        type='AV2GlobalDiffusionDataset',
        static_root=STATIC_ROOT,
        rendered_gt_root=RENDERED_GT_ROOT,
        semantic_root=SEMANTIC_ROOT,
        scene_list=SCENE_LIST,
        use_condition=_use_condition,
        mix_with_08_prob=_mix_prob,
        M=M,
        num_queries=NUM_QUERIES,
        class_budget=CLASS_BUDGET,
        drop_instance=(mode == 'denoise'),
        load_image=(mode == 'denoise'),
        cond_max_side=1024 if mode == 'denoise' else None,
        cond_fixed_size=(1024, 1024) if mode == 'denoise' else None,
        pad_fill='gaussian',
        pad_sigma=0.2,
        seed=1234,
        proposal_root=_proposal_root,
    )


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=make_data('denoise'),
    # True validation dataset roots
    val=dict(
        type='AV2GlobalDiffusionDataset',
        static_root=VAL_STATIC_ROOT,
        rendered_gt_root=VAL_RENDERED_GT_ROOT,
        semantic_root=VAL_SEMANTIC_ROOT,
        scene_list=None,
        use_condition='10',
        mix_with_08_prob=0.0,
        M=M,
        num_queries=NUM_QUERIES,
        class_budget=CLASS_BUDGET,
        drop_instance=False,
        load_image=True,
        cond_max_side=1024,
        cond_fixed_size=(1024, 1024),
        pad_fill='gaussian',
        pad_sigma=0.2,
        seed=1234,
        proposal_root=None,
    ),
    test=dict(
        type='AV2GlobalDiffusionDataset',
        static_root=VAL_STATIC_ROOT,
        rendered_gt_root=VAL_RENDERED_GT_ROOT,
        semantic_root=VAL_SEMANTIC_ROOT,
        scene_list=None,
        use_condition='10',
        mix_with_08_prob=0.0,
        M=M,
        num_queries=NUM_QUERIES,
        class_budget=CLASS_BUDGET,
        drop_instance=False,
        load_image=True,
        cond_max_side=1024,
        cond_fixed_size=(1024, 1024),
        pad_fill='gaussian',
        pad_sigma=0.2,
        seed=1234,
    ),
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-5)
runner = dict(type='EpochBasedRunner', max_epochs=1)
evaluation = dict(interval=1, metric='chamfer')
work_dir = './work_dirs/av2_polydiffuse_official_base'
