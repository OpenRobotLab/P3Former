# dataset settings
dataset_type = 'MNuScenesDataset'
#data_root = '/home/PJLAB/xiaozeqi/Desktop/git/Cylinder3D_mmdet3d/data/sequences'
data_root = '/mnt/petrelfs/share_data/zhangjingwei/datasets/nuscenes'
label_mapping = "configs/seg/label_mapping/nuscenes.yaml"
class_names = ['noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
'terrain', 'manmade', 'vegetation']
#point_cloud_range = [0, -40, -3, 70.4, 40, 1]

input_modality = dict(use_lidar=True, use_camera=False)

#dbinfos 存放了从样本中单独提出的物体，便于在样本包含物体较少时增强样本
# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'kitti_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
#     classes=class_names,
#     sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6))

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#    backend='petrel', path_mapping=dict({'data/nuscenes':'s3://openmmlab/datasets/detection3d/nuscenes'}))
# file_client_args = None

load_from_dir=True

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MyLoadAnnotations3D',
        label_mapping=label_mapping,
        with_seg_3d=True,
        load_type='panoptic',
        datatype='nuscenes',
        file_client_args=file_client_args,
        seg_3d_dtype = 'np.uint8'),

    dict(type='DefaultFormatBundle3D', class_names=class_names),#to tensor
    dict(type='Collect3D', keys=['points','pts_semantic_mask', 'pts_instance_mask'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        # a wrapper in order to successfully call test function
        # actually we don't perform test-time-aug
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # dict(
            #     type='GlobalRotScaleTrans',
            #     rot_range=[0, 0],
            #     scale_ratio_range=[1., 1.],
            #     translation_std=[0, 0, 0]),
            # dict(
            #     type='RandomFlip3D',
            #     sync_2d=False,
            #     flip_ratio_bev_horizontal=0.0,
            #     flip_ratio_bev_vertical=0.0),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
    # dict(type='DefaultFormatBundle3D', class_names=class_names),#to tensor
    # dict(type='Collect3D', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=4,
    #     use_dim=4,
    #     file_client_args=file_client_args),
    dict(
        type='MyLoadAnnotations3D',
        label_mapping=label_mapping,
        with_seg_3d=True,
        load_type='panoptic',
        datatype='nuscenes',
        file_client_args=file_client_args,
        seg_3d_dtype = 'np.uint8'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['pts_semantic_mask','pts_instance_mask'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            imageset="/mnt/petrelfs/share_data/zhangjingwei/datasets/nuscenes_mini/nuscenes_infos_train.pkl",
            version='v1.0-trainval',
            # imageset="data/nuscenes/nuscenes_infos_mini.pkl",
            # version='v1.0-mini',
            task='panoptic',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            label_mapping=label_mapping,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            )),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        #ann_file=data_root + 'kitti_infos_val.pkl',
        # imageset="data/nuscenes/nuscenes_infos_val.pkl",
        version='v1.0-trainval',
        imageset="/mnt/petrelfs/share_data/zhangjingwei/datasets/nuscenes_mini/nuscenes_infos_val.pkl",
        # version='v1.0-mini',
        task='panoptic',
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        label_mapping=label_mapping,
        ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        #ann_file=data_root + 'kitti_infos_val.pkl',
        # imageset="data/nuscenes/nuscenes_infos_test.pkl",
        # version='v1.0-trainval',
        imageset="/mnt/petrelfs/share_data/zhangjingwei/datasets/nuscenes_mini/nuscenes_infos_val.pkl",
        version='v1.0-mini',
        task='panoptic',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        label_mapping=label_mapping,
        ))

evaluation = dict(interval=1, pipeline=eval_pipeline,task='panoptic')

