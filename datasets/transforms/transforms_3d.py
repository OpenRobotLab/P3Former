# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose, RandomResize, Resize
from mmdet.datasets.transforms import (PhotoMetricDistortion, RandomCrop,
                                       RandomFlip)
from mmengine import is_list_of, is_tuple_of

from mmdet3d.models.task_modules import VoxelGenerator
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                                LiDARInstance3DBoxes)
from mmdet3d.structures.ops import box_np_ops
from mmdet3d.structures.points import BasePoints
from mmdet3d.datasets.transforms.data_augment_utils import noise_per_object_v3_


@TRANSFORMS.register_module(force=True)
class _PolarMix(BaseTransform):
    """PolarMix data augmentation.

    The polarmix transform steps are as follows:

        1. Another random point cloud is picked by dataset.
        2. Exchange sectors of two point clouds that are cut with certain
           azimuth angles.
        3. Cut point instances from picked point cloud, rotate them by multiple
           azimuth angles, and paste the cut and rotated instances.

    Required Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)
    - dataset (:obj:`BaseDataset`)

    Modified Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)

    Args:
        instance_classes (List[int]): Semantic masks which represent the
            instance.
        swap_ratio (float): Swap ratio of two point cloud. Defaults to 0.5.
        rotate_paste_ratio (float): Rotate paste ratio. Defaults to 1.0.
        pre_transform (Sequence[dict], optional): Sequence of transform object
            or config dict to be composed. Defaults to None.
        prob (float): The transformation probability. Defaults to 1.0.
    """

    def __init__(self,
                 instance_classes: List[int],
                 swap_ratio: float = 0.5,
                 rotate_paste_ratio: float = 1.0,
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        assert is_list_of(instance_classes, int), \
            'instance_classes should be a list of int'
        self.instance_classes = instance_classes
        self.swap_ratio = swap_ratio
        self.rotate_paste_ratio = rotate_paste_ratio

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def polar_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            mix_results (dict): Mixed dict picked from dataset.

        Returns:
            dict: output dict after transformation.
        """
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']

        mix_panoptic = False
        if 'pts_instance_mask' in mix_results:
            mix_instance_mask = mix_results['pts_instance_mask']
            mix_instance_mask += (1000<<16) # not overlap id
            pts_instance_mask = input_dict['pts_instance_mask']
            mix_panoptic = True

        # 1. swap point cloud
        if np.random.random() < self.swap_ratio:
            start_angle = (np.random.random() - 1) * np.pi  # -pi~0
            end_angle = start_angle + np.pi
            # calculate horizontal angle for each point
            yaw = -torch.atan2(points.coord[:, 1], points.coord[:, 0])
            mix_yaw = -torch.atan2(mix_points.coord[:, 1], mix_points.coord[:,
                                                                            0])

            # select points in sector
            idx = (yaw <= start_angle) | (yaw >= end_angle)
            mix_idx = (mix_yaw > start_angle) & (mix_yaw < end_angle)

            # swap
            points = points.cat([points[idx], mix_points[mix_idx]])
            pts_semantic_mask = np.concatenate(
                (pts_semantic_mask[idx.numpy()],
                 mix_pts_semantic_mask[mix_idx.numpy()]),
                axis=0)
            
            if mix_panoptic:
                pts_instance_mask = np.concatenate(
                    (pts_instance_mask[idx.numpy()],
                    mix_instance_mask[mix_idx.numpy()]),
                    axis=0)                

        # 2. rotate-pasting
        if np.random.random() < self.rotate_paste_ratio:
            # extract instance points
            instance_points, instance_pts_semantic_mask = [], []
            if mix_panoptic:
                instance_pts_instance_mask = []
            for instance_class in self.instance_classes:
                mix_idx = mix_pts_semantic_mask == instance_class
                instance_points.append(mix_points[mix_idx])
                instance_pts_semantic_mask.append(
                    mix_pts_semantic_mask[mix_idx])
                if mix_panoptic:
                    instance_pts_instance_mask.append(mix_instance_mask[mix_idx])
            instance_points = mix_points.cat(instance_points)
            instance_pts_semantic_mask = np.concatenate(
                instance_pts_semantic_mask, axis=0)
            if mix_panoptic:
               instance_pts_instance_mask = np.concatenate(
                instance_pts_instance_mask, axis=0) 

            # rotate-copy
            copy_points = [instance_points]
            copy_pts_semantic_mask = [instance_pts_semantic_mask]
            if mix_panoptic:
                copy_pts_instance_mask = [instance_pts_instance_mask]
            angle_list = [
                np.random.random() * np.pi * 2 / 3,
                (np.random.random() + 1) * np.pi * 2 / 3
            ]
            for angle in angle_list:
                new_points = instance_points.clone()
                new_points.rotate(angle)
                copy_points.append(new_points)
                copy_pts_semantic_mask.append(instance_pts_semantic_mask)
                if mix_panoptic:
                    copy_pts_instance_mask.append(instance_pts_instance_mask)
            copy_points = instance_points.cat(copy_points)
            copy_pts_semantic_mask = np.concatenate(
                copy_pts_semantic_mask, axis=0)
            if mix_panoptic:
                copy_pts_instance_mask = np.concatenate(
                copy_pts_instance_mask, axis=0)

            points = points.cat([points, copy_points])
            pts_semantic_mask = np.concatenate(
                (pts_semantic_mask, copy_pts_semantic_mask), axis=0)
            if mix_panoptic:
                pts_instance_mask = np.concatenate(
                (pts_instance_mask, copy_pts_instance_mask), axis=0)

        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        if mix_panoptic:
            input_dict['pts_instance_mask'] = pts_instance_mask
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through PolarMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before polarmix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.polar_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(instance_classes={self.instance_classes}, '
        repr_str += f'swap_ratio={self.swap_ratio}, '
        repr_str += f'rotate_paste_ratio={self.rotate_paste_ratio}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module(force=True)
class _LaserMix(BaseTransform):
    """LaserMix data augmentation.

    The lasermix transform steps are as follows:

        1. Another random point cloud is picked by dataset.
        2. Divide the point cloud into several regions according to pitch
           angles and combine the areas crossly.

    Required Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)
    - dataset (:obj:`BaseDataset`)

    Modified Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)

    Args:
        num_areas (List[int]): A list of area numbers will be divided into.
        pitch_angles (Sequence[float]): Pitch angles used to divide areas.
        pre_transform (Sequence[dict], optional): Sequence of transform object
            or config dict to be composed. Defaults to None.
        prob (float): The transformation probability. Defaults to 1.0.
    """

    def __init__(self,
                 num_areas: List[int],
                 pitch_angles: Sequence[float],
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        assert is_list_of(num_areas, int), \
            'num_areas should be a list of int.'
        self.num_areas = num_areas

        assert len(pitch_angles) == 2, \
            'The length of pitch_angles should be 2, ' \
            f'but got {len(pitch_angles)}.'
        assert pitch_angles[1] > pitch_angles[0], \
            'pitch_angles[1] should be larger than pitch_angles[0].'
        self.pitch_angles = pitch_angles

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def laser_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """LaserMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            mix_results (dict): Mixed dict picked from dataset.

        Returns:
            dict: output dict after transformation.
        """
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']

        rho = torch.sqrt(points.coord[:, 0]**2 + points.coord[:, 1]**2)
        pitch = torch.atan2(points.coord[:, 2], rho)
        pitch = torch.clamp(pitch, self.pitch_angles[0] + 1e-5,
                            self.pitch_angles[1] - 1e-5)

        mix_rho = torch.sqrt(mix_points.coord[:, 0]**2 +
                             mix_points.coord[:, 1]**2)
        mix_pitch = torch.atan2(mix_points.coord[:, 2], mix_rho)
        mix_pitch = torch.clamp(mix_pitch, self.pitch_angles[0] + 1e-5,
                                self.pitch_angles[1] - 1e-5)

        num_areas = np.random.choice(self.num_areas, size=1)[0]
        angle_list = np.linspace(self.pitch_angles[1], self.pitch_angles[0],
                                 num_areas + 1)
        out_points = []
        out_pts_semantic_mask = []

        mix_panoptic = False
        if 'pts_instance_mask' in mix_results:
            mix_instance_mask = mix_results['pts_instance_mask']
            mix_instance_mask += (1000<<16) # not overlap id
            pts_instance_mask = input_dict['pts_instance_mask']
            out_pts_instance_mask = []
            mix_panoptic = True

        for i in range(num_areas):
            # convert angle to radian
            start_angle = angle_list[i + 1] / 180 * np.pi
            end_angle = angle_list[i] / 180 * np.pi
            if i % 2 == 0:  # pick from original point cloud
                idx = (pitch > start_angle) & (pitch <= end_angle)
                out_points.append(points[idx])
                out_pts_semantic_mask.append(pts_semantic_mask[idx.numpy()])
                if mix_panoptic:
                    out_pts_instance_mask.append(pts_instance_mask[idx.numpy()])
            else:  # pickle from mixed point cloud
                idx = (mix_pitch > start_angle) & (mix_pitch <= end_angle)
                out_points.append(mix_points[idx])
                out_pts_semantic_mask.append(
                    mix_pts_semantic_mask[idx.numpy()])
                if mix_panoptic:
                    out_pts_instance_mask.append(mix_instance_mask[idx.numpy()])
        out_points = points.cat(out_points)
        out_pts_semantic_mask = np.concatenate(out_pts_semantic_mask, axis=0)
        input_dict['points'] = out_points
        input_dict['pts_semantic_mask'] = out_pts_semantic_mask

        if mix_panoptic:
            out_pts_instance_mask = np.concatenate(out_pts_instance_mask, axis=0)
            input_dict['pts_instance_mask'] = out_pts_instance_mask
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """LaserMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through LaserMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before lasermix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.laser_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_areas={self.num_areas}, '
        repr_str += f'pitch_angles={self.pitch_angles}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str
