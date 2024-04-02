# optimizer
optimizer = dict(type='AdamW', lr=0.0005)
# max_norm=10 is better for SECOND
#optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict()

lr_config = dict(
    policy='step',
    warmup=None,
    min_lr=1e-4,#权宜之计
    step=25
    )
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=40)
