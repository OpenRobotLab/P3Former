# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from mmseg.core import add_prefix
from mmdet3d.models.builder import build_backbone, build_head, build_loss, build_neck, build_voxel_encoder, SEGMENTORS
from mmdet3d.models.segmentors.base import Base3DSegmentor
from p3former.ops.voxel.voxelize import SphericalVoxelization
import time


@SEGMENTORS.register_module()
class CylinderPanoptic(Base3DSegmentor):
    """3D Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be thrown during inference.
    """

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 #middle_encoder=None,
                 backbone,
                 is_fix_backbone,
                 decode_head=None,
                 ignore_index=0,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 visual=False,
                 count=False,
                 ensemble=False,
                 semantic_branch=False,
                 semantic_head=None,
                 pred_semantic=False, 
                 use_sem=False,
                 direct_sem=False,
                 ):
        super(CylinderPanoptic, self).__init__(init_cfg=init_cfg)
        self.is_fix_backbone = is_fix_backbone
        self.voxel_layer = SphericalVoxelization(**voxel_layer)
        self.voxel_encoder = build_voxel_encoder(voxel_encoder)
        self.backbone = build_backbone(backbone)\

        self.use_sem = use_sem

        if neck is not None:
            self.neck = build_neck(neck)
        if decode_head is not None:
            self.decode_head = build_head(decode_head)# TODO if can insert cfg
        else:
            self.decode_head = None

        self.visual = visual
        self.count = count
        self.ensemble = ensemble
        self.semantic_branch = semantic_branch
        if self.semantic_branch:
            self.semantic_head = build_head(semantic_head)
        self.pred_semantic = pred_semantic
        self.direct_sem = direct_sem
            
    def encode_decode(self):
        pass
    def extract_feat(self):
        pass
    def forward_train(self, points, img_metas, pts_semantic_mask, pts_instance_mask, valid=None):
        if valid is None:
            valid = [valid]*len(points)
        if self.is_fix_backbone:
            with torch.no_grad():
                data_batch = self.voxel_layer([points,pts_semantic_mask,pts_instance_mask,valid])
                coords, features_3d = self.voxel_encoder(data_batch['pt_fea'],data_batch['grid'])
                feature, _ = self.backbone(features_3d, coords, len(data_batch['grid']))
        else:
            data_batch = self.voxel_layer([points,pts_semantic_mask,pts_instance_mask,valid])
            coords, features_3d = self.voxel_encoder(data_batch['pt_fea'],data_batch['grid'])
            feature, _ = self.backbone(features_3d, coords, len(data_batch['grid']))
        
        losses = dict()
        if self.decode_head is not None:
            losses.update(self.decode_head.forward_train(feature,data_batch,len(img_metas)))
        if self.semantic_branch:
            losses.update(self.semantic_head.forward_train(feature,data_batch,len(img_metas)))
        return losses

    def inference(self,points, img_metas, rescale=None, pts_semantic_mask=None, pts_instance_mask=None, valid=None):
        if valid is None:
            valid = [valid]*len(points)
        if pts_semantic_mask is not None:
            points = [points]
            data_batch = self.voxel_layer([points,pts_semantic_mask,pts_instance_mask,valid],aug=True) 
        else:
            points = points          
            data_batch = self.voxel_layer([points])
        coords, features_3d = self.voxel_encoder(data_batch['pt_fea'],data_batch['grid'])
        feature, _ = self.backbone(features_3d, coords, len(data_batch['grid']))
        if self.pred_semantic:
            seg_prob = self.semantic_head.forward_test(feature)
            seg_map = seg_prob.argmax(1).cpu()
            pred_sem_preds_list=[]
            for i in range(seg_map.shape[0]):
                pred_sem_preds_list.append(seg_map[i,data_batch['grid'][i][:,0],data_batch['grid'][i][:,1],data_batch['grid'][i][:,2]])
            ins_id_list = [None]
            return pred_sem_preds_list, ins_id_list, None, None
        else:
            if self.use_sem:
                seg_prob = self.semantic_head.forward_test(feature)
                seg_map = seg_prob.argmax(1)
                sembranch_list=[]
                for i in range(seg_map.shape[0]):
                    sembranch_list.append(seg_map[i,data_batch['grid'][i][:,0],data_batch['grid'][i][:,1],data_batch['grid'][i][:,2]])
            else:
                sembranch_list=None
            pred_sem_preds_list, ins_id_list = self.decode_head.forward_test(feature,data_batch,len(points),data_batch['grid'],
                                                                                    pred_sem=sembranch_list, ensemble = self.ensemble)
            if pts_semantic_mask is not None and pts_instance_mask is not None:
                return pred_sem_preds_list, ins_id_list, pts_semantic_mask, pts_instance_mask
            else:
                return pred_sem_preds_list, ins_id_list, None, None

    def simple_test(self, points, img_metas, rescale=True, pts_semantic_mask=None, pts_instance_mask=None, **kwargs):

        pt_sem_preds_list, ins_id_list, _, _= self.inference(points, img_metas, rescale, pts_semantic_mask, pts_instance_mask, **kwargs)

        results = dict()
        results['semantic_mask'] = pt_sem_preds_list[0]
        results['ins_ids'] = ins_id_list[0]
        if pts_semantic_mask is not None:
            results['pts_semantic_mask'] = pts_semantic_mask[0].cpu()
        if pts_instance_mask is not None:
            results['pts_instance_mask'] = pts_instance_mask[0].cpu()

        return [results]

    def aug_test(self, points, img_metas, rescale=True):
        #TODO only support one sample now
        pt_sem_preds_list, ins_id_list = self.inference(points, img_metas, rescale)
        pt_sem_preds_list, ins_id_list = self.merge_aug(pt_sem_preds_list, ins_id_list, img_metas)
        
        results = dict()
        results['semantic_mask'] = pt_sem_preds_list
        results['ins_ids'] = ins_id_list
        return [results]
    
    def merge_aug(pt_sem_preds_list, ins_id_list, img_metas):
        for iter in range(len(img_metas)):
            if img_metas[iter]['pcd_horizontal']:
                pt_sem_preds_list[i]
