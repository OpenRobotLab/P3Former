_base_ = [
    '../_base_/datasets/semantickitti_panoptic_lpmix.py', '../_base_/models/p3former.py',
    '../_base_/default_runtime.py'
]

# optimizer
# This schedule is mainly used by models on nuScenes dataset

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

model = dict(
    voxel_encoder=dict(
        feat_channels=[64, 128, 256, 256],
        in_channels=6,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(
        input_channels=16,
        base_channels=32,
        more_conv=True,
        out_channels=256),
    decode_head=dict(
        num_decoder_layers=6,
        num_queries=128,
        embed_dims=256,
        cls_channels=(256, 256, 20),
        mask_channels=(256, 256, 256, 256, 256),
        thing_class=[0,1,2,3,4,5,6,7],
        stuff_class=[8,9,10,11,12,13,14,15,16,17,18],
        ignore_index=19
    ))


lr = 0.0008
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 32],
        gamma=0.2)
]

train_dataloader = dict(batch_size=2, )

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))

custom_imports = dict(
    imports=[
        'p3former.backbones.cylinder3d',
        'p3former.data_preprocessors.data_preprocessor',
        'p3former.decode_heads.p3former_head',
        'p3former.segmentors.p3former',
        'p3former.task_modules.samplers.mask_pseduo_sampler',
        'evaluation.metrics.panoptic_seg_metric',
        'datasets.semantickitti_dataset',
        'datasets.transforms.loading',
        'datasets.transforms.transforms_3d',
    ],
    allow_failed_imports=False)
