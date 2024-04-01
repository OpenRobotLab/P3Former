# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import tempfile
import warnings
from os import path as osp
import os
from torch.utils import data
import yaml
import torch

from mmdet3d.core import show_result, show_seg_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet.datasets import DATASETS
from mmseg.datasets import DATASETS as SEG_DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from p3former.datasets.mycustom_3d_seg import MyCustom3DSegDataset
from mmdet3d.datasets.pipelines import Compose
from mmcv.utils import print_log
from p3former.utils.eval.pan_eval import pan_eval

import pickle

from nuscenes import NuScenes

@DATASETS.register_module()
@SEG_DATASETS.register_module()
class MNuScenesDataset(MyCustom3DSegDataset):

    def __init__(self,
                 data_root,
                 version,
                 task='lidarseg',
                 imageset="train",
                 label_mapping="nuscenes.yaml",
                 return_ref=True,
                 pipeline=None,
                 classes=None,
                 palette=None,
                 modality=None,
                 test_mode=False,
                 ignore_index=0,
                 scene_idxs=None,):

        nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        self.data_root=data_root
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.semkittiyaml = semkittiyaml
        self.task = task

        self.epoch = 0
        self.max_miou_epoch = 0
        self.max_mpq_epoch = 0
        self.max_miou = 0
        self.max_mpq = 0


        self.return_ref = return_ref

        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.data_infos = data['data_list']
        self.data_path = data_root
        self.nusc = nusc

        super().__init__(
            data_root=data_root,
            ann_file=None,
            pipeline=pipeline,
            classes=classes,
            palette=palette,
            modality=modality,
            test_mode=test_mode,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs)
          
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        #pts_semantic_mask_path = info['pts_filename'].replace('velodyne', 'labels')[:-3] + 'label'
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        anns_results = dict()
        if self.task == 'semantic':
            pts_semantic_mask_path = os.path.join(self.nusc.dataroot,
                                                    self.nusc.get('lidarseg', lidar_sd_token)['filename']) #根据lidar token 找到 filename
            anns_results['pts_semantic_mask_path']=pts_semantic_mask_path
        if self.task == 'panoptic':
            pts_panoptic_mask_path = os.path.join(self.nusc.dataroot,
                                                self.nusc.get('panoptic', lidar_sd_token)['filename']) #根据lidar token 找到 filename
            anns_results['pts_panoptic_mask_path']=pts_panoptic_mask_path
        return anns_results

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        raise NotImplementedError('should not be used')
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=False,
                with_seg_3d=True),
            dict(
                type='PointSegClassMapping',
                valid_cat_ids=self.VALID_CLASS_IDS,
                max_cat_id=np.max(self.ALL_CLASS_IDS)),
            dict(
                type='DefaultFormatBundle3D',
                with_label=False,
                class_names=self.CLASSES),
            dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
        ]
        return Compose(pipeline)

    
    def get_classes_and_palette(self, classes=None, palette=None):
            class_names = classes

            self.label_map = self.learning_map

            self.label2cat = {
                    i: cat_name
                    for i, cat_name in enumerate(class_names)
                }
            return None, None

    
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        lidar_path = self.data_infos[0]['lidar_points']['lidar_path'][16:]
        info = self.data_infos[index]
        lidar_path = info['lidar_points']['lidar_path']
        pts_filename = os.path.join(self.data_path, 'samples/LIDAR_TOP', lidar_path)
        input_dict = dict(
            pts_filename=pts_filename)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        return input_dict
    
    def evaluate(self,
                results,
                task,
                metric=None,
                logger=None,
                show=False,
                out_dir=None,
                pipeline=None,):
            """Evaluate.

            Evaluation in semantic segmentation protocol.

            Args:
                results (list[dict]): List of results.
                metric (str | list[str]): Metrics to be evaluated.
                logger (logging.Logger | None | str): Logger used for printing
                    related information during evaluation. Defaults to None.
                show (bool, optional): Whether to visualize.
                    Defaults to False.
                out_dir (str, optional): Path to save the visualization results.
                    Defaults to None.
                pipeline (list[dict], optional): raw data loading for showing.
                    Default: None.

            Returns:
                dict: Evaluation results.
            """
            from mmdet3d.core.evaluation import seg_eval
            from mmcv.runner import get_dist_info
            if not isinstance(
                results, list):
                print("...")
                return None
            # assert isinstance(
            #     results, list), '...'# f'Expect results to be list, got {type(results)}.'
            assert len(results) > 0, 'Expect length of results > 0.'
            assert len(results) == len(self.data_infos)
            assert isinstance(
                results[0], dict
            ), f'Expect elements in results to be dict, got {type(results[0])}.'

            load_pipeline = self._get_pipeline(pipeline)

            if task == 'semantic':
                if isinstance(results[0]['semantic_mask'],list):
                    pred_sem_masks = []
                    for result in results:
                        for s in result['semantic_mask']:
                            pred_sem_masks.append(s)
                else:
                    pred_sem_masks = [result['semantic_mask'] for result in results]
                gt_sem_masks = [
                    self._extract_data(
                        i, load_pipeline, 'pts_semantic_mask', load_annos=True)
                    for i in range(len(self.data_infos))
                ]# 迎合mmdetection3d
                ret_dict = seg_eval(
                    gt_sem_masks,
                    pred_sem_masks,
                    self.label2cat,
                    self.ignore_index,
                    logger=logger)

                if show:
                    self.show(pred_sem_masks, out_dir, pipeline=pipeline)

                self.epoch+=1
                if ret_dict['miou']>self.max_miou:
                    self.max_miou_epoch = self.epoch
                    self.max_miou=ret_dict['miou']
                print_log('\n max_miou_epoch:' + str(self.max_miou_epoch), logger=logger)    
                print_log('\n max_miou:' + str(self.max_miou), logger=logger)

            elif task == 'panoptic':
                if isinstance(results[0]['semantic_mask'],list):# not fixed 
                    pred_sem_masks = []
                    pred_ins_ids = []
                    for result in results:
                        for s in result['semantic_mask']:
                            pred_sem_masks.append(s.numpy().reshape(-1))
                        for i in result['ins_ids']:
                            pred_ins_ids.append(i.numpy().reshape(-1))
                else:
                    pred_sem_masks = [result['semantic_mask'].numpy().reshape(-1) for result in results]
                    pred_ins_ids = [result['ins_ids'].numpy().reshape(-1) for result in results]
                
                if results[0]['pts_semantic_mask'] is not None and results[0]['pts_instance_mask'] is not None:
                    pts_semantic_mask = [result['pts_semantic_mask'].numpy().reshape(-1) for result in results]
                    pts_instance_mask = [result['pts_instance_mask'].numpy().reshape(-1) for result in results]
                    pts_name = [data_info['token'] for data_info in self.data_infos]
                    
                else:
                    data_tuples = [
                        self._extract_data(
                            i, load_pipeline, ['pts_semantic_mask', 'pts_instance_mask'], load_annos=True)
                        for i in range(len(self.data_infos))
                    ]

                    pts_semantic_mask = [data_tuple[0] for data_tuple in data_tuples]
                    pts_instance_mask = [data_tuple[1] for data_tuple in data_tuples]
                    pts_name = [data_info['token'] for data_info in self.data_infos]


                print(len(pts_semantic_mask))
                print(len(pred_sem_masks))
                count = 0 
                for i in range(len(pts_semantic_mask)):
                    if pts_semantic_mask[i].shape == pred_sem_masks[i].shape:
                        continue
                    count +=1
                print('number:'+str(count))
                ret_dict = pan_eval(
                    pred_sem_masks=pred_sem_masks,
                    pred_ins_ids=pred_ins_ids,
                    pts_semantic_mask=pts_semantic_mask,
                    pts_instance_mask=pts_instance_mask,
                    dataset='nuscenes',
                    pcd_fname=pts_name,
                    logger=logger) 

                self.epoch+=1
                if ret_dict['miou']>self.max_miou:
                    self.max_miou_epoch = self.epoch
                    self.max_miou=ret_dict['miou']
                if ret_dict['mpq']>self.max_mpq:
                    self.max_mpq_epoch = self.epoch
                    self.max_mpq=ret_dict['mpq']
                print_log('\n max_miou_epoch:' + str(self.max_miou_epoch), logger=logger)    
                print_log('\n max_miou:' + str(self.max_miou), logger=logger)
                print_log('\n max_mpq_epoch:' + str(self.max_mpq_epoch), logger=logger)  
                print_log('\n max_mpq:' + str(self.max_mpq), logger=logger) 
            
            elif task == 'debug':
                if isinstance(results[0]['semantic_mask'],list):
                    pred_sem_masks = []
                    pred_ins_ids = []
                    for result in results:
                        for s in result['semantic_mask']:
                            pred_sem_masks.append(s.numpy().reshape(-1))
                        for i in result['ins_ids']:
                            pred_ins_ids.append(i.numpy().reshape(-1))
                else:
                    pred_sem_masks = [result['semantic_mask'].numpy().reshape(-1) for result in results]
                    pred_ins_ids = [result['ins_ids'].numpy().reshape(-1) for result in results]

                data_tuples = [
                    self._extract_data(
                        i, load_pipeline, ['pts_semantic_mask', 'pts_instance_mask', 'points'], load_annos=True)
                    for i in range(len(self.data_infos))
                ]

                pts_semantic_mask = [data_tuple[0] for data_tuple in data_tuples]#TODO eval_pipeline no cuda?
                pts_instance_mask = [data_tuple[1] for data_tuple in data_tuples]
                pts_name = [data_info['pts_filename'] for data_info in self.data_infos]

                points = [data_tuple[2] for data_tuple in data_tuples]

                ret_dict = pan_eval(
                    pred_sem_masks,
                    pred_ins_ids,
                    pts_semantic_mask,
                    pts_instance_mask,
                    pts_name,
                    logger=logger,
                    points=points)

            else: raise NotImplementedError

            return ret_dict

    def test(self,
                results,
                task,
                savename,
                taskset,
                metric=None,
                logger=None,
                show=False,
                out_dir=None,
                pipeline=None,
                ):
            """Evaluate.

            Evaluation in semantic segmentation protocol.

            Args:
                results (list[dict]): List of results.
                metric (str | list[str]): Metrics to be evaluated.
                logger (logging.Logger | None | str): Logger used for printing
                    related information during evaluation. Defaults to None.
                show (bool, optional): Whether to visualize.
                    Defaults to False.
                out_dir (str, optional): Path to save the visualization results.
                    Defaults to None.
                pipeline (list[dict], optional): raw data loading for showing.
                    Default: None.

            Returns:
                dict: Evaluation results.
            """
         

            if not isinstance(
                results, list):
                print("...")
                return None

            if task == 'semantic':
                pred_sem_masks = [result['semantic_mask'].numpy() for result in results]
                self.save_test_results(pred_sem_masks, savename, 'lidarseg', taskset)



            elif task == 'panoptic':
                pred_sem_masks = [result['semantic_mask'].numpy().reshape(-1) for result in results]
                pred_ins_ids = [result['ins_ids'].numpy().reshape(-1) for result in results]
                self.save_test_results(pred_sem_masks, savename=savename, task='panoptic', taskset=taskset, pred_ins_ids=pred_ins_ids)

    def save_test_results(self, pred_sem_masks, savename, task, taskset, pred_ins_ids=None, output_dir='output/'):
        results_dir = os.path.join(output_dir, savename)
        lidar_dir = os.path.join(results_dir, task, taskset)
        meta_dir = os.path.join(results_dir, taskset)
        if not os.path.exists(lidar_dir):
            os.makedirs(lidar_dir, exist_ok=True)
        if not os.path.exists(meta_dir):
            os.makedirs(meta_dir, exist_ok=True)

        if task=='panoptic':     
            meta =  {"meta": {
                    "task": "segmentation",
                    "use_camera": False,
                    "use_lidar": True,
                    "use_radar": False,
                    "use_map": False,
                    "use_external": False}}
        else:   
            meta =  {"meta": {"use_camera": False,
                    "use_lidar": True,
                    "use_radar": False,
                    "use_map": False,
                    "use_external": False}}
        
        import json
        output = open(os.path.join(meta_dir, 'submission.json'), 'w')
        json_meta = json.dumps(meta)
        output.write(json_meta)
        output.close()

        for i in range(len(self.data_infos)):
            if pred_ins_ids is None:
                sem_preds = pred_sem_masks[i].astype(np.uint8)
                #sem_inv = class_inv_lut[sem_preds].astype(np.uint8)
            else:
                sem_preds = (pred_ins_ids[i] + pred_sem_masks[i]*1000).astype(np.uint16)

            lidar_sd_token = self.nusc.get('sample', self.data_infos[i]['token'])['data']['LIDAR_TOP']
            if task=='panoptic':
                bin_file_path = lidar_sd_token + "_panoptic.npz"
            else:
                bin_file_path = lidar_sd_token + "_lidarseg.bin"
            np.savez_compressed(os.path.join(lidar_dir, bin_file_path), data=sem_preds)

    def read_sem(self, root):
        pred_names = []
        pred_paths = os.path.join(root, "sequences",
                                "08", "predictions")
        # populate the label names
        seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
        seq_pred_names.sort()
        pred_names.extend(seq_pred_names)

        pred_list = []
        for pred_file in pred_names:

            pred = np.fromfile(pred_file, dtype=np.int32)
            pred = pred.reshape((-1))    # reshape to vector
            pred = pred & 0xFFFF         # get lower half for semantics
            pred = np.vectorize(self.learning_map.__getitem__)(pred)
            pred_list.append(pred)
            if len(pred_list)%200==0:
                print(len(pred_list))
            
        return pred_list