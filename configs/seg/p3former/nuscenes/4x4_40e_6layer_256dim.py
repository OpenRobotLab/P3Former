_base_ = [
    '../../_base_/schedules/40e_schedule.py',
    '../../_base_/models/cylinder_base.py',
    '../../_base_/datasets/NuScenes_panoptic.py',
    '../../_base_/default_runtime.py'
]

grid_size = [480,360,32]
norm_cfg = dict(type='BN1d', eps=1e-5, momentum=0.01)
task = 'panoptic'
init_size = 16
num_proposals = 128
num_classes = 17
num_stuff_classes = 6 # 不包含0
num_thing_classes = 11
feature_dim = 256
num_decoder_layers = 6
is_fix_backbone = False
resume_from = None
checkpoint = None
job_name = '.'
zero_keep = False
thing_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

model = dict(
    type='CylinderPanoptic',
    voxel_layer=dict(
        grid_size=grid_size,
        point_cloud_range=[0, '-np.pi', -5, 50, 'np.pi', 3],
        thing_list=thing_list,
        use_polarmix=True,
    ),
    voxel_encoder=dict(
        grid_size=grid_size,
        norm_cfg=norm_cfg
    ),
    backbone=dict(
        output_shape=grid_size,
        norm_cfg=norm_cfg,
        init_size=init_size,
    ),
    decode_head=dict(
        type='SimplePanopticHead',
        fist_layer_cfg=dict(
            type='PosFirstLayer',
            splitpanoptic=True,
            sem_layer_cfg=dict(
                type='SemLayer',
                num_thing_classes=num_thing_classes,
            ),
            init_layer_cfg=dict(
                type='InitLayer',
                norm_cfg=norm_cfg,
                embed_dims=feature_dim,
                num_classes=num_classes,
                is_more_conv=True,
                num_proposals=num_proposals,
                is_bias=False,
                grid_size=grid_size,
                postype='pol_xyz',
                conv_after_simple_pos=True,
                splitpanoptic=True,
                point_cloud_range=[0, '-np.pi', -5, 50, 'np.pi', 3],
            ), 
            pred_layer_cfg=dict(
                type='PredictLayer',
                in_channels=feature_dim,
                out_channels=feature_dim,
                num_classes=num_classes,
                num_cls_fcs=1,
                num_mask_fcs=3,
                num_pos_fcs=2,
                off_candidate_thr=0.6,
                act_cfg=dict(type='GELU'),
                pred_coor_feat_mask=True,
            ),
        ),
        iter_layer_cfg=dict(
            type='PosIterLayer',
            pred_layer_cfg=dict(
                type='PredictLayer',
                in_channels=feature_dim,
                out_channels=feature_dim,
                num_classes=num_classes,
                num_cls_fcs=1,
                num_mask_fcs=3,
                num_pos_fcs=2,
                off_candidate_thr=0.6,
                act_cfg=dict(type='GELU'),
                pred_coor_feat_mask=True,
            ),
            update_layer_cfg=dict(
                type='UpdatorLayer',
                updator_cfg=dict(
                    type='Updator',
                    in_channels=feature_dim,
                    feat_channels=feature_dim,
                    out_channels=feature_dim,
                    act_cfg=dict(type='GELU'),
                    norm_cfg=dict(type='LN'),),
                in_channels=feature_dim,
                conv_kernel_size=1,
                num_heads=8,
                with_ffn=True,
                feedforward_channels=2048,
                num_ffn_fcs=2,
                dropout=0.0,
                ffn_act_cfg=dict(type='GELU')#TODO no GeLU?
            ),
        ),
        loss_layer_cfg=dict(
            type='SetLossLayer',
            assign_layer_cfg=dict(
                type='SplitPrevAssignLayer',
                num_classes=num_classes,
                num_stuff_classes=num_stuff_classes,
                num_thing_classes=num_thing_classes,
                assigner=dict(                
                    type='MyMaskHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', gamma=4.0,alpha=0.25,weight=1.0),
                    dice_cost=dict(type='DiceCost', weight=2.0, pred_act=True),
                    mask_cost=dict(type='BinaryFocalLossCost', gamma=2.0, alpha=0.25, weight=1.0)
                ),
                sampler=dict(type='MyMaskPseudoSampler')
              
            ),
            loss_layer_cfg=dict(
                type='LossLayer',
                num_classes=num_classes,
                loss_mask=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    reduction='mean',
                    loss_weight=1.0),
                loss_dice=dict(
                    type='DiceLoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=4.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_offset=None,
                loss_center=None,
                show_offset=False,
                show_center=True,
                loss_sem=True,
                loss_coor_dice_weight=0.2
            ),
        ),
        gt_layer_cfg=dict(
            type='GtConvertLayer',
            task=task,
            thing_class=thing_list,
            point_cloud_range=[0, '-np.pi', -5, 50, 'np.pi', 3],
        ),
        getpan_layer_cfg=dict(
            type='NewGetPanopticLayer',
            num_thing_classes=num_thing_classes,
            num_proposals=num_proposals+num_stuff_classes,
            num_stuff_classes=num_stuff_classes,
            iou_thr=0.8,
            classnum=num_classes,
            ignore_class=[0],
            thing_class=thing_list,
            object_mask_thr=0.4,
        ),
        num_decoder_layers=num_decoder_layers,
        use_prev=True,
    ),
    is_fix_backbone=is_fix_backbone,
)

data_root = 'data/nuscenes/'
samples_per_gpu = 4
times = 1

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=samples_per_gpu,
    train=dict(
        times=times,
        dataset=dict(
            data_root = data_root,
            imageset=data_root+"nuscenes_infos_train.pkl",
            version='v1.0-trainval',
        )
    ),
    test=dict(
        data_root = data_root,
        imageset=data_root+"nuscenes_infos_val.pkl",
        version='v1.0-trainval',
    ),
    val=dict(
        data_root = data_root,
        imageset=data_root+"nuscenes_infos_val.pkl",
        version='v1.0-trainval',
    ),
)

evaluation_interval = 2
evaluation = dict(
    interval=evaluation_interval,
)

log_interval = 50
log_config = dict(
    interval=log_interval
)

optimizer = dict(
    lr = 0.0005
)

runner = dict(type='EpochBasedRunner', max_epochs=200)

custom_imports = dict(
    imports=[
        'p3former.models.backbones.Asymm_3d_spconv',
        'p3former.models.decode_heads.simple_pan_head',
        'p3former.models.decode_heads.updator',
        'p3former.models.decode_heads.iter_head',
        'p3former.models.segmentors.cylinderpanoptic',
        'p3former.models.voxel_encoders.cylinder_encoder',
        'p3former.ops.voxel.voxelize',
        'p3former.datasets.mynuscenes_dataset',
        'p3former.datasets.pipelines.loading',
        'p3former.utils.mask_hungarian_assigner',
        'p3former.utils.mask_pseudo_sampler',
        'p3former.utils.position_encoding'
    ],
    allow_failed_imports=False)
