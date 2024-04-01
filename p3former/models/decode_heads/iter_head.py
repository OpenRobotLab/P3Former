from cmath import nan
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import (FFN, MultiheadAttention,
                                         build_transformer_layer)
from mmdet.models.utils import build_transformer
from mmdet.models.utils.builder import TRANSFORMER
import spconv
from mmdet.core import multi_apply

@TRANSFORMER_LAYER.register_module()
class BaseFirstLayer(nn.Module):
    def __init__(self,
                init_layer_cfg, 
                pred_layer_cfg,
                ):
        super(BaseFirstLayer, self).__init__()
        self.first_layer = build_transformer_layer(init_layer_cfg)
        self.pred_layer = build_transformer_layer(pred_layer_cfg)

    def forward(self, feature, bs):
        feature_split, proposal_feat_stack, = self.first_layer(feature, bs)
        logits, sigmoid_masks_list = self.pred_layer(feature_split, proposal_feat_stack)

        pred_dict = dict()
        pred_dict['logits'] = logits

        middle_dict = dict()
        middle_dict['sigmoid_masks_list'] = sigmoid_masks_list
        middle_dict['proposal_feat_stack'] = proposal_feat_stack
        middle_dict['feature_split'] = feature_split
        
        return middle_dict, pred_dict

@TRANSFORMER_LAYER.register_module()
class BaseIterLayer(nn.Module):
    def __init__(self,
                pred_layer_cfg,
                update_layer_cfg,
                ):
        super(BaseIterLayer, self).__init__()
        self.pred_layer = build_transformer_layer(pred_layer_cfg)
        self.update_layer = build_transformer_layer(update_layer_cfg)

    def forward(self, input_dict):
        sigmoid_masks_list = input_dict['sigmoid_masks_list']
        proposal_feat_stack = input_dict['proposal_feat_stack']
        feature_split = input_dict['feature_split']

        proposal_feat_stack = self.update_layer(sigmoid_masks_list, feature_split, proposal_feat_stack)
        mask_feat_list, sigmoid_masks_list= \
                        self.pred_layer(feature_split, proposal_feat_stack)
        
        pred_dict = dict()
        pred_dict['logits'] = mask_feat_list

        middle_dict = dict()
        middle_dict['sigmoid_masks_list'] = sigmoid_masks_list
        middle_dict['proposal_feat_stack'] = proposal_feat_stack
        middle_dict['feature_split'] = feature_split

        return middle_dict, pred_dict

@TRANSFORMER_LAYER.register_module()
class PosFirstLayer(BaseFirstLayer):
    def __init__(self,
                init_layer_cfg, 
                pred_layer_cfg,
                sem_layer_cfg=None,
                splitpanoptic=False,
                ):
        super(PosFirstLayer, self).__init__(init_layer_cfg, 
                                            pred_layer_cfg,)
        self.splitpanoptic = splitpanoptic
        if sem_layer_cfg is not None:
            self.sem_layer = build_transformer_layer(sem_layer_cfg)

    def forward(self, feature, bs, centerlist=None, data_batch=None):
        feature_split, indice_split, proposal_feat, featurewoindice_split, con_seg_stack, coor_feature_list = \
                                                                self.first_layer(feature, bs, centerlist, data_batch)
        if self.splitpanoptic:
            sem_pred, proposal_feat = self.sem_layer(feature_split, con_seg_stack, proposal_feat)

        pred_dict = self.pred_layer(feature_split, indice_split, proposal_feat, featurewoindice_split, coor_feature_list)
        
        pred_dict['indice_split'] = indice_split
        if self.splitpanoptic:
            pred_dict['sem_pred'] = sem_pred
        

        middle_dict = dict()
        middle_dict['sigmoid_masks_list'] = pred_dict['sigmoid_masks']
        middle_dict['proposal_feat'] = proposal_feat
        middle_dict['feature_split'] = feature_split
        middle_dict['featurewoindice_split'] = featurewoindice_split
        middle_dict['indice_split'] = indice_split
        middle_dict['coor_feature_list'] = coor_feature_list
        
        return middle_dict, pred_dict

@TRANSFORMER_LAYER.register_module()
class PosIterLayer(BaseIterLayer):
    def __init__(self,
                pred_layer_cfg,
                update_layer_cfg,
                indice_update_layer_cfg=None,
                ):
        super(PosIterLayer, self).__init__(pred_layer_cfg,
                                            update_layer_cfg,)
        self.indice_update_layer_cfg = indice_update_layer_cfg
        if indice_update_layer_cfg is not None:
            self.indice_update_layer = build_transformer_layer(indice_update_layer_cfg)


    def forward(self, input_dict, centerlist=None):
        sigmoid_masks_list = input_dict['sigmoid_masks_list']
        proposal_feat = input_dict['proposal_feat']
        feature_split = input_dict['feature_split']
        featurewoindice_split = input_dict['featurewoindice_split']
        indice_split = input_dict['indice_split']
        coor_feature_list = input_dict['coor_feature_list']

        proposal_feat = self.update_layer(sigmoid_masks_list, feature_split, proposal_feat, indice_split, featurewoindice_split=featurewoindice_split)
        pred_dict = self.pred_layer(feature_split, indice_split, proposal_feat, featurewoindice_split, coor_feature_list)
        pred_dict['indice_split'] = indice_split

        middle_dict = dict()
        middle_dict['sigmoid_masks_list'] = pred_dict['sigmoid_masks']
        middle_dict['proposal_feat'] = proposal_feat
        middle_dict['feature_split'] = feature_split
        middle_dict['featurewoindice_split'] = featurewoindice_split
        middle_dict['indice_split'] = indice_split
        middle_dict['coor_feature_list'] = coor_feature_list

        return middle_dict, pred_dict

@TRANSFORMER_LAYER.register_module()
class SetLossLayer(nn.Module):
    def __init__(self,
                assign_layer_cfg,
                loss_layer_cfg,
                ):
        super(SetLossLayer, self).__init__()

        self.assign_layer = build_transformer_layer(assign_layer_cfg)
        self.loss_layer = build_transformer_layer(loss_layer_cfg)


    def forward(self, pred_dict, target_dict, layer, prev_pred_dict=None,data_batch=None,zero_layer_pred=None):
        if layer == 0:
            pred_dict_assign, target_dict_assign, weight_dict_assign = self.assign_layer(pred_dict, target_dict, layer=layer)
            losses = self.loss_layer(pred_dict_assign, target_dict_assign, weight_dict_assign, layer, data_batch=data_batch,zero_layer_pred=zero_layer_pred)
        else:
            pred_dict_assign, target_dict_assign, weight_dict_assign = self.assign_layer(pred_dict, target_dict, prev_pred_dict, layer)
            losses = self.loss_layer(pred_dict_assign, target_dict_assign, weight_dict_assign, layer, data_batch=data_batch,zero_layer_pred=zero_layer_pred)

        return losses

