grid_shape = [480, 360, 32]
model = dict(
    type='_P3Former',
    data_preprocessor=dict(
        type='_Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2],
            max_num_points=-1,
            max_voxels=-1,
        ),
    ),
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=6,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(
        type='_Asymm3DSpconv',
        grid_size=grid_shape,
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)),
    decode_head=dict(
        type='_P3FormerHead',
        num_classes=20,
        num_queries=128,
        embed_dims=128,
        point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2],
        assigner_zero_layer_cfg=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                        dict(type='mmdet.FocalLossCost', weight=1.0, binary_input=True, gamma=2.0, alpha=0.25),
                        dict(type='mmdet.DiceCost', weight=2.0, pred_act=True),
                    ]),
        assigner_cfg=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                        dict(type='mmdet.FocalLossCost', gamma=4.0,alpha=0.25,weight=1.0),
                        dict(type='mmdet.FocalLossCost', weight=1.0, binary_input=True, gamma=2.0, alpha=0.25),
                        dict(type='mmdet.DiceCost', weight=2.0, pred_act=True),
                    ]),
        sampler_cfg=dict(type='_MaskPseudoSampler'),
        loss_mask=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_dice=dict(type='mmdet.DiceLoss', loss_weight=2.0),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=4.0,
            alpha=0.25,
            loss_weight=1.0),
    ),
    train_cfg=None,
    test_cfg=dict(mode='whole'),
)
