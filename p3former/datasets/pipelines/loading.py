# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import yaml
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.datasets.pipelines import LoadAnnotations3D

@PIPELINES.register_module()
class MyLoadAnnotations3D(LoadAnnotations3D):
    def __init__(self,
                label_mapping,
                 with_bbox_3d=False,
                 with_label_3d=False,
                 with_seg_3d=False,
                 load_type='panoptic',
                 seg_3d_dtype='int',
                 datatype='semantickitti',
                
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox_3d=with_bbox_3d,
            with_label_3d=with_label_3d,
            with_seg_3d=with_seg_3d,
            file_client_args=file_client_args)
        self.seg_3d_dtype = seg_3d_dtype
        self.datatype = datatype

        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.loadtype=load_type

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=eval(self.seg_3d_dtype)).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, eval(self.seg_3d_dtype))

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results
    
    def _load_panoptic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_panoptic_mask_path = results['ann_info']['pts_panoptic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_panoptic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=eval(self.seg_3d_dtype)).copy()['data']
        except:
            mmcv.check_file_exist(pts_panoptic_mask_path)
            pts_panoptic_mask = np.load(pts_panoptic_mask_path)['data']

        results['pts_panoptic_mask'] = pts_panoptic_mask
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        
        if self.loadtype == 'panoptic':
            if self.datatype == 'semantickitti':
                results = self._load_semantic_seg_3d(results)
                annotated_data = results['pts_semantic_mask']
                ins_labels = annotated_data
                sem_labels = annotated_data & 0xFFFF
                valid = np.isin(sem_labels, self.things_ids).reshape(-1) # use 0 to filter out valid indexes is enough
                from mmcv.parallel import DataContainer as DC
                import torch
                results['valid'] = DC(torch.from_numpy(valid))
            else:
                #results = self._load_semantic_seg_3d(results)
                results = self._load_panoptic_seg_3d(results)
                sem_labels = results['pts_panoptic_mask'] // 1000 
                ins_labels = results['pts_panoptic_mask']


            sem_labels = np.vectorize(self.learning_map.__getitem__)(sem_labels)

            results['pts_semantic_mask'] = sem_labels.astype(np.long)
            results['pts_instance_mask'] = ins_labels.astype(np.long)

        elif self.loadtype == 'semantic':
            results = self._load_semantic_seg_3d(results)
            annotated_data = results['pts_semantic_mask']
            if self.datatype == 'semantickitti':
                sem_labels = annotated_data & 0xFFFF
            else:
                sem_labels = annotated_data % 1000
            sem_labels = np.vectorize(self.learning_map.__getitem__)(sem_labels)
            results['pts_semantic_mask'] = sem_labels.astype(np.long)

        
        return results




