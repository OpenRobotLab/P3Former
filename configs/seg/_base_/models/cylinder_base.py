grid_size = [10,10,5]
fea_dim = 9
out_fea_dim = 256 #all use 256?
num_input_features = 16
use_norm = True
init_size = 32
is_fix_backbone = False

norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.01)

model = dict(
    type='CylinderPanoptic',
    voxel_layer=dict(
        point_cloud_range=[0, '-np.pi', -4, 50, 'np.pi', 2],
        grid_size=grid_size,
    ),
    voxel_encoder=dict(
        type='CylinderVFE',
        grid_size=grid_size,
        fea_dim=fea_dim,
        out_fea_dim=out_fea_dim,
        fea_compre=num_input_features,
        norm_cfg=norm_cfg
    ),
    #middle_encoder=dict(),
    backbone=dict(
        type='Asymm_3d_spconv',
        output_shape=grid_size,
        use_norm=use_norm,
        num_input_features=num_input_features,
        init_size=init_size,
        norm_cfg=norm_cfg
    ),
    is_fix_backbone = is_fix_backbone
)

# custom_imports = dict(
#     imports=[
#         'cylinder.models.backbones.Asymm_3d_spconv',
#         'cylinder.models.segmentors.cylinderpanoptic',
#         'cylinder.models.voxel_encoders.cylinder_encoder',
#         'cylinder.ops.voxel.voxelize',
#         'cylinder.datasets.mysemantickitti_dataset',
#         'cylinder.datasets.pipelines.loading',
#         'cylinder.datasets.mysemantickitti_dataset',
#         'cylinder.datasets.pipelines.loading',
#     ],
#     allow_failed_imports=False)
