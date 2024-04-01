# Copyright (c) OpenMMLab. All rights reserved.
from os import truncate
from mmdet3d.models.builder import build_loss
from torch import nn as nn
import torch
from mmdet.models import HEADS
from mmcv.runner import BaseModule
try:
    import spconv.pytorch as spconv
except ImportError:
    import spconv
from mmcv.cnn import ConvModule, xavier_init
import torch.nn.functional as F

from p3former.models.losses.lovasz_loss import flatten_probas

from mmcv.cnn.bricks.transformer import (FFN, MultiheadAttention,
                                         build_transformer_layer)
from mmdet.models.utils import build_transformer
from mmcv.cnn import (ConvModule, bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from copy import deepcopy

from mmdet.core import  reduce_mean, build_assigner, build_sampler, multi_apply

from mmdet.models.losses import accuracy
import numpy as np

@HEADS.register_module()
class SimplePanopticHead(BaseModule):

    def __init__(self,
                fist_layer_cfg,
                iter_layer_cfg,
                loss_layer_cfg,
                gt_layer_cfg,
                num_decoder_layers,
                use_prev,
                getpan_layer_cfg=None,
                zero_keep=True,
                visual=False,
                visual_layer_cfg=dict(type='VisualizeLayer',
                                            grid_size=[480,360,32,128],
                                            visual_layer=0,
                                            cls_num=20,),
                assign_layer_cfg=dict(
                    type='PrevAssignLayer',
                    num_classes=20,
                    assigner=dict(                
                        type='MyMaskHungarianAssigner',
                        cls_cost=dict(type='FocalLossCost', gamma=4.0,alpha=0.25,weight=1.0),
                        dice_cost=dict(type='DiceCost', weight=20000.0, pred_act=True),
                        mask_cost=dict(type='BinaryFocalLossCost', gamma=2.0, alpha=0.25, weight=1.0)
                    ),
                    sampler=dict(type='MyMaskPseudoSampler')
                ),
                tongji=False,
                point_wise=False,
                with_point_loss=False,
                count=False,
                use_sem=False,
                pred_sem=False,
                use_gt_sem=False,
                noise_rate=0.3,
                ):

        super(SimplePanopticHead, self).__init__()

        self.num_decoder_layers = num_decoder_layers

        self.fist_layer = build_transformer_layer(fist_layer_cfg)
        self.loss_layer = build_transformer_layer(loss_layer_cfg)
        self.gt_layer = build_transformer_layer(gt_layer_cfg)
        self.use_prev = use_prev
        self.zero_keep = zero_keep
        self.point_wise = point_wise
        self.pred_sem = pred_sem

        self.use_sem = use_sem
        self.use_gt_sem = use_gt_sem

        self.noise_rate = noise_rate

        self.getpan_layer = build_transformer_layer(getpan_layer_cfg)

        self.with_point_loss = with_point_loss

        self.iter_layers = nn.ModuleList()
        for l in range(self.num_decoder_layers):
            if l == self.num_decoder_layers-1:
                iter_layer_cfg['pred_layer_cfg']['last_layer']=True
            self.iter_layers.append(build_transformer_layer(iter_layer_cfg))
        self.assign_layer_cfg = assign_layer_cfg

    def init_weights(self):
        """Initialize weights of transformer decoder in GroupFree3DHead."""
        # initialize transformer
        for m in self.parameters():
            if m.dim() > 1:
                xavier_init(m, distribution='uniform')


    def forward(self, feature, bs, data_batch=None, test_mode=False):
        if 'pt_valid' in data_batch:
            target_dict=self.gt_layer.convert(data_batch) #! for hack
        else:
            target_dict=dict()

        if 'sparse_sem_labels' in data_batch:
            if self.point_wise:
                target_dict['sparse_sem_labels'] = data_batch['pt_labs']
            else:
                target_dict['sparse_sem_labels'] = data_batch['sparse_sem_labels']

        pred_dict_list = []
        middle_dict, pred_dict = self.fist_layer(feature, bs,data_batch=data_batch)
        pred_dict_list.append(pred_dict)

        for iter_layer in self.iter_layers:
            middle_dict, pred_dict = iter_layer(middle_dict)
            pred_dict_list.append(pred_dict)         

        return pred_dict_list,  target_dict

    def loss(self,
            pred_dict_list,
            data_batch,
            inputs=None,
            target_dict=None):
        losses = dict()

        if self.use_prev: # use preditions from previous layer for assignment
            for i in range(self.num_decoder_layers+1):
                if i==0:
                    losses.update(self.loss_layer(pred_dict_list[i],target_dict,i,data_batch=data_batch,zero_layer_pred=pred_dict_list[0]))
                else:
                    losses.update(self.loss_layer(pred_dict_list[i],target_dict,i,pred_dict_list[i-1],data_batch=data_batch,zero_layer_pred=pred_dict_list[0]))
        else:
            for i in range(self.num_decoder_layers+1):
                losses.update(self.loss_layer(pred_dict_list[i],target_dict,i))
        
        if self.with_point_loss:
            pt_sem_preds_list, pt_sem_weights_list = self.getpan_layer.get_semlogits(pred_dict_list[-1]['labels'], pred_dict_list[-1]['masks'])
            loss_lovasz = 0
            loss_ce = 0

            loss_l=build_loss(dict(type='Lovasz_Softmax',))
            loss_c = nn.CrossEntropyLoss(ignore_index=0)

            gt_sem = data_batch['sparse_sem_labels']
            valid_bs = 0
            for en, (ps,ts) in enumerate(zip(pt_sem_preds_list,gt_sem)):

                assert (ts.max()<=19) & (ts.min()>=0)
                if ps.sum()==0 or ts.sum()==0:
                    continue
                
                # pw = pt_sem_weights_list[en]
                # valid_voxel = pw>0.5
                # ts[~valid_voxel] = 0 #这会导致置信度下降

                valid_bs +=1

                loss_lovasz += loss_l(nn.functional.softmax(ps,dim=1), ts, ignore=0)
                loss_ce += loss_c(ps, ts)

            if valid_bs>0:
                losses['loss_lovasz'] = loss_lovasz/valid_bs
                losses['loss_ce'] = loss_ce/valid_bs
            else:
                losses['loss_lovasz'] = ps.sum()*0.0
                losses['loss_ce'] = ps.sum()*0.0

        return losses     

    def forward_train(self, inputs, 
                        data_batch, 
                        bs=None,):# label_tensor can be voxel or point or voxel_sparse
        pred_dict_list, target_dict = self.forward(inputs, bs, data_batch)
        losses = self.loss(pred_dict_list, data_batch, inputs, target_dict)
        return losses

    def forward_test(self, inputs, data_batch, bs=None, points=None, ensemble=False, pred_sem=None):
        pred_dict_list, target_dict = self.forward(inputs, bs, data_batch, test_mode=True)
        pred_logits = pred_dict_list[-1]['labels']
        pt_mask_list = pred_dict_list[-1]['masks']
        pred_mask = pt_mask_list
        pt_sem_preds_list, ins_id_list = self.getpan_layer.get_panoptic(pred_logits, pred_mask, points, pred_dict_list=pred_dict_list, target_dict=target_dict)

        point_semantic_preds, point_instance_ids  = self.generate_point_predictions(inputs, data_batch['grid'], pt_sem_preds_list, ins_id_list)

        return point_semantic_preds, point_instance_ids

    def generate_point_predictions(self, sparse_voxels, point_grid_indices,
                                   semantic_preds, instance_ids):
        """Get point-wise predictions.

        Args:
            sparse_voxels (SparseConvTensor): Sparse voxels including
                features and grids.
            point_grid_indices (list[torch.Tensor]): Grid indices of
                point clouds.
            semantic_preds (list[torch.Tensor]): Semantic predictions of
                                                                    voxels.
            instance_ids (list[torch.Tensor]): Instance predictions of voxels.

        Returns:
            tuple[list[torch.Tensor]]: Semantic predictions and instance
                predicions of points.
        """
        semantic_preds = torch.cat(semantic_preds, 0)
        instance_ids = torch.cat(instance_ids, 0)
        panoptic_results = semantic_preds + (instance_ids << 8)
        sparse_voxels.features = panoptic_results[:, None]
        dense_feature = sparse_voxels.dense()
        assert not torch.any(torch.isnan(dense_feature))
        point_semantic_preds = []
        point_instance_ids = []
        for batch_i, grid_ind_i in enumerate(point_grid_indices):
            point_panoptic_results = dense_feature[batch_i, :, grid_ind_i[:,
                                                                          0],
                                                   grid_ind_i[:, 1],
                                                   grid_ind_i[:, 2]].squeeze(0)
            point_semantic_preds.append(point_panoptic_results & 0xFF)
            point_instance_ids.append(point_panoptic_results >> 8)
        return (point_semantic_preds, point_instance_ids)