from mmcv import Config

from global_diffusion_map.plugin.datasets.av2_diffusion_dataset import AV2GlobalDiffusionDataset  # noqa: F401

_base_ = ['../_base_/default_runtime.py']

# Use PolyDiffuse MapTR plugin
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# Paths (defaults for AV2 oldsplit train)
STATIC_ROOT = 'maptracker/work_dirs/static_gt_vector_simp/av2_oldsplit/train'
RENDERED_GT_ROOT = 'maptracker/work_dirs/rendered_gt/av2_oldsplit/train'
SEMANTIC_ROOT = 'maptracker/work_dirs/semantic/train'
SCENE_LIST = None  # or a text file of scene ids

# Geometry and caps
point_cloud_range = [-1.0, -1.0, -2.0, 1.0, 1.0, 2.0]  # normalized space for training loss
# PolyDiffuse-aligned defaults
M = 20
NUM_QUERIES = 50

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 128
bev_w_ = 128

map_classes = ['divider', 'ped_crossing', 'boundary']
num_map_classes = len(map_classes)

model = dict(
    type='MapTR',
    use_grid_mask=False,
    video_test_mode=False,
    img_backbone=dict(
        type='ResNet', depth=18, num_stages=4, out_indices=(3,), frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False), norm_eval=True, style='pytorch'),
    img_neck=dict(
        type='FPN', in_channels=[512], out_channels=_dim_, start_level=0,
        add_extra_convs='on_output', num_outs=_num_levels_, relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='MapTRHead', bev_h=bev_h_, bev_w=bev_w_, num_vec=NUM_QUERIES, num_pts_per_vec=M,
        num_pts_per_gt_vec=M, dir_interval=1, query_embed_type='instance_pts', transform_method='minmax',
        gt_shift_pts_pattern='v2', num_classes=num_map_classes, in_channels=_dim_, sync_cls_avg_factor=True,
        with_box_refine=True, as_two_stage=False, code_size=2, code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='MapTRPerceptionTransformer', num_cams=1, rotate_prev_bev=False, use_shift=False,
            use_can_bus=False, embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder', num_layers=1, pc_range=point_cloud_range, num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[dict(type='TemporalSelfAttention', embed_dims=_dim_, num_levels=1)],
                    feedforward_channels=_ffn_dim_, ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='MapTRDecoder', num_layers=3, pc_range=point_cloud_range, dataset_type='nuscenes',
                return_intermediate=True, embed_dims=_dim_, num_heads=8, num_levels=1, real_h=bev_h_, real_w=bev_w_,
                timestep_embed=128 * 4,
                transformerlayers=dict(
                    type='PolyDetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1),
                        dict(type='MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1),
                        dict(type='CustomMSDeformableAttention', embed_dims=_dim_, num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_, ffn_dropout=0.1,
                    operation_order=(
                        'self_attn', 'norm', 'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        bbox_coder=dict(type='MapTRNMSFreeCoder', post_center_range=[-2, -2, -2, 2, 2, 2, 2, 2],
                        pc_range=point_cloud_range, max_num=NUM_QUERIES, voxel_size=[0.1, 0.1, 4],
                        num_classes=num_map_classes),
        positional_encoding=dict(type='LearnedPositionalEncoding', num_feats=_pos_dim_,
                                 row_num_embed=bev_h_, col_num_embed=bev_w_),
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


def make_data(split: str, mode: str):
    # mode: 'guide' or 'denoise'
    return dict(
        type='AV2GlobalDiffusionDataset',
        static_root=STATIC_ROOT,
        rendered_gt_root=RENDERED_GT_ROOT,
        semantic_root=SEMANTIC_ROOT,
        scene_list=SCENE_LIST,
        use_condition='11' if mode == 'denoise' else '11',
        mix_with_08_prob=0.0,
        M=M,
        num_queries=NUM_QUERIES,
        drop_instance=(mode == 'denoise'),
        load_image=(mode == 'denoise'),
        seed=1234,
    )


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=make_data('train', mode='denoise'),
    val=make_data('val', mode='denoise'),
    test=make_data('val', mode='denoise'),
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-5)
runner = dict(type='EpochBasedRunner', max_epochs=1)
evaluation = dict(interval=1, metric='chamfer')
work_dir = './work_dirs/av2_polydiffuse'
