_base_ = [
    '../_base_/datasets/semantickitti_lpmix.py', '../_base_/models/cylinder3d.py',
    '../_base_/default_runtime.py'
]

# optimizer
# This schedule is mainly used by models on nuScenes dataset
lr = 0.008
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    # max_norm=10 is better for SECOND
    clip_grad=dict(max_norm=10, norm_type=2))

# training schedule for 2x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 32],
        gamma=0.1)
]