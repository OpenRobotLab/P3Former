from cmath import nan
from selectors import EpollSelector
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import (FFN, MultiheadAttention,
                                         build_transformer_layer,
                                         build_positional_encoding,
                                         build_attention)
from mmdet.models.utils import build_transformer
from mmdet.models.utils.builder import TRANSFORMER
import spconv

def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, indice_key=indice_key)
                     
from mmdet3d.models.builder import build_loss
from mmdet.core import  reduce_mean, build_assigner, build_sampler, multi_apply
from mmdet.models.losses import accuracy
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn import Conv2d, build_plugin_layer, kaiming_init
import copy

import numpy as np
import copy

def pol2continue(indice_split):
    for idx in range(len(indice_split)):
        tmp = indice_split[idx]
        tmask = tmp[:,1]>0.5
        tmp[tmask,1] = 1 - tmp[tmask,1]
    return indice_split

def pol2cat(indice_split, point_cloud_range=[0, -np.pi, -4, 50, np.pi, 2]):
    point_cloud_range = torch.Tensor(point_cloud_range)

    cat_indice_split = []
    for idx in range(len(indice_split)):
        indice = indice_split[idx].clone()
        for i in range(3):
            indice[:,i] = indice[:,i]*(point_cloud_range[i+3]-point_cloud_range[i]) - \
                                                                        - point_cloud_range[i]
        x = indice[:,0] * torch.cos(indice[:,1])
        y = indice[:,0] * torch.sin(indice[:,1])
        cat_indice = torch.stack([x, y, indice[:,2]], 1)                                                                
        cat_indice = cat_indice / (point_cloud_range[3]-point_cloud_range[0])
        #cat_indice = cat_indice*0.5+0.5 #[0,1]
        #assert ((cat_indice<=1) & (cat_indice>=0)).sum()==(cat_indice.shape[0]*cat_indice.shape[1])
        cat_indice_split.append(cat_indice)
    return cat_indice_split

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@TRANSFORMER.register_module()
class Updator(nn.Module):
    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 gate_sigmoid=True,
                 gate_norm_act=False,
                 activate_out=False,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN')):
        super(Updator, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out,
                                     1)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        if self.gate_norm_act:
            self.gate_norm = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, update_feature, input_feature):
        # update_feature from feature!
        update_feature = update_feature.reshape(-1, self.in_channels)
        num_proposals = update_feature.size(0)

        # update_feature: in_channels
        parameters = self.dynamic_layer(update_feature)


        param_in = parameters[:, :self.num_params_in].view(
            -1, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels)

        input_feats = self.input_layer(
            input_feature.reshape(num_proposals, -1, self.feat_channels))
        input_in = input_feats[..., :self.num_params_in]
        input_out = input_feats[..., -self.num_params_out:]

        gate_feats = input_in * param_in.unsqueeze(-2)
        if self.gate_norm_act:
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            input_gate = input_gate.sigmoid()
            update_gate = update_gate.sigmoid()
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out:
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        features = update_gate * param_out.unsqueeze(
            -2) + input_gate * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features

@TRANSFORMER_LAYER.register_module()
class InitLayer(nn.Module):
    def __init__(self,
                norm_cfg,
                embed_dims,
                is_more_conv,
                grid_size,
                num_classes=None,
                num_thing_classes=None,
                num_proposals=20,
                is_bias=False, 
                postype='xysin',
                conv_after_simple_pos=False,
                pos_dim=3,
                point_wise=False,
                using_high=False,
                splitpanoptic=False,
                input_dims=64,
                point_cloud_range=None,
                ):
        super(InitLayer, self).__init__()

        self.is_more_conv = is_more_conv
        self.embed_dims = embed_dims
        self.num_proposals = num_proposals
        self.is_bias = is_bias
        self.grid_size = torch.tensor(grid_size).double().cuda()
        self.postype = postype

        self.pos_dim = pos_dim
        self.conv_after_simple_pos = conv_after_simple_pos

        if input_dims is None:
            input_dims = embed_dims

        self.point_wise = point_wise
        self.splitpanoptic = splitpanoptic
        self.point_cloud_range = point_cloud_range

        if point_cloud_range[1]=='-np.pi':
            point_cloud_range[1]=-np.pi
            point_cloud_range[4]=np.pi

        self.kernel_proposal = spconv.SubMConv3d(embed_dims, num_proposals, indice_key="logit", 
                                kernel_size=1, stride=1, padding=0, bias=is_bias)

                                    
        if self.postype is 'pol' or self.postype is 'xyz':
            self.position_proj = nn.Linear(pos_dim, self.embed_dims)
            self.position_norm = build_norm_layer(dict(type='LN'), self.embed_dims)[1]
        if self.postype is 'pol_xyz':
            self.pol_proj = nn.Linear(pos_dim, self.embed_dims)
            self.pol_norm = build_norm_layer(dict(type='LN'), self.embed_dims)[1]
            self.xyz_proj = nn.Linear(pos_dim, self.embed_dims)
            self.xyz_norm = build_norm_layer(dict(type='LN'), self.embed_dims)[1]
            if self.conv_after_simple_pos:
                self.more_after_conv = nn.ModuleList()
                self.more_after_conv.append(
                    nn.Linear(self.embed_dims, self.embed_dims, bias=False))
                self.more_after_conv.append(
                    build_norm_layer(dict(type='LN'), self.embed_dims)[1])
                self.more_after_conv.append(build_activation_layer(dict(type='ReLU', inplace=True),))                          

        if self.is_more_conv:
            self.addConv = conv3x3(input_dims, embed_dims, indice_key='mc')
            self.addBn = build_norm_layer(norm_cfg, embed_dims)[1]
            self.addAct = nn.LeakyReLU()
        
        if self.conv_after_simple_pos:
            self.after_conv = nn.ModuleList()
            self.after_conv.append(
                nn.Linear(self.embed_dims, self.embed_dims, bias=False))
            self.after_conv.append(
                build_norm_layer(dict(type='LN'), self.embed_dims)[1])
            self.after_conv.append(build_activation_layer(dict(type='ReLU', inplace=True),))

        if self.splitpanoptic:
            self.conv_seg = nn.Conv3d(embed_dims, num_classes, kernel_size=1, stride=1, padding=0, bias=is_bias)

    def forward(self, feature, bs, center_split=None, data_batch=None): # center_split for hack
        if self.is_more_conv:
            feature = self.addConv(feature)
            feature.features = self.addBn(feature.features)
            feature.features = self.addAct(feature.features)

        input_feature = feature.features #[V,C]

        queries = self.kernel_proposal.weight.clone().squeeze(0).squeeze(0).repeat(bs,1,1).permute(0,2,1) #[B,N,C] K=1
        if self.splitpanoptic:
            conv_seg = self.conv_seg.weight.clone().squeeze(-1).squeeze(-1).repeat(1,1,bs).permute(2,0,1)
        else:
            conv_seg = None

        feature_split = self.batch_split(input_feature,feature.indices,bs)
        featurewoindice_split = self.batch_split(input_feature,feature.indices,bs)
        indice_split =  self.batch_split(feature.indices,feature.indices,bs)
        indice_split = [i[:,1:]/self.grid_size  for i in indice_split]


        pos_encoding_list = None
        if self.postype is 'pol':
            pos_encoding_list = []
            for i in range(bs):
                pos_encoding = self.position_norm(self.position_proj(indice_split[i][:,:self.pos_dim].float()))
                if self.conv_after_simple_pos:
                    for after_conv in self.after_conv:
                        feature_split[i] = after_conv(feature_split[i])
                        pos_encoding = after_conv(pos_encoding)
                        pos_encoding_list.append(pos_encoding)
                feature_split[i] = feature_split[i] + pos_encoding

        elif self.postype is 'xyz':
            pos_encoding_list = []
            if not self.point_wise:
                cat_indice = pol2cat(indice_split)
            else:
                cat_indice = indice_split
            for i in range(bs):
                dd = cat_indice[i][:,:self.pos_dim]
                pos_encoding = self.position_norm(self.position_proj(dd.float()))
                feature_split[i] = feature_split[i] + pos_encoding
                if self.conv_after_simple_pos:
                    for after_conv in self.after_conv:
                        feature_split[i] = after_conv(feature_split[i])
                        pos_encoding = after_conv(pos_encoding)
                        pos_encoding_list.append(pos_encoding)
                feature_split[i] = feature_split[i] + pos_encoding

            indice_split = cat_indice

        elif self.postype is 'pol_xyz':
            cat_indice = pol2cat(indice_split, self.point_cloud_range)
            pos_encoding_list = []

            indice_split = pol2continue(indice_split) 

            for i in range(bs):
                pos_encoding_xyz = self.xyz_norm(
                    self.xyz_proj(cat_indice[i][:, :self.pos_dim].float()))
                pos_encoding_pol = self.pol_norm(
                    self.pol_proj(indice_split[i][:, :self.pos_dim].float()))
                if self.conv_after_simple_pos: # check if necessary
                    for after_conv in self.more_after_conv:
                        feature_split[i] = after_conv(feature_split[i])
                        pos_encoding_xyz = after_conv(pos_encoding_xyz)
                        pos_encoding_pol = after_conv(pos_encoding_pol)
                pos_encoding = pos_encoding_xyz + pos_encoding_pol
                feature_split[i] = feature_split[i] + pos_encoding
                feature_split[i] = feature_split[i].float()
                pos_encoding_list.append(pos_encoding)

            indice_split = cat_indice                                                                  

        return feature_split, indice_split, queries, featurewoindice_split, conv_seg, pos_encoding_list
    
    def batch_split(self,input_feature,indices,bs):
        feature_batch = []
        for i in range(bs):
            feature_batch.append(input_feature[indices[:,0]==i])
        return feature_batch    

@TRANSFORMER_LAYER.register_module()      
class SemLayer(nn.Module):
    def __init__(self,
            num_thing_classes,
            ):
        super(SemLayer, self).__init__()
        self.num_thing_classes = num_thing_classes


    def forward_single(self, feature, conv_seg, queries):
        sem_pred = torch.einsum("nc,vc->vn", conv_seg, feature)
        stuff_queries = conv_seg[self.num_thing_classes:].clone()
        queries = torch.cat([queries, stuff_queries], dim=0)
        return sem_pred, queries


    def forward(self, feature, conv_seg, queries):
        sem_pred, queries = multi_apply( self.forward_single, 
                                                feature,
                                                conv_seg,
                                                queries)
        return sem_pred, queries

@TRANSFORMER_LAYER.register_module()
class PredictLayer(nn.Module):
    def __init__(self,
                in_channels=128,
                out_channels=128,
                num_classes=20,
                num_things_classes=8,
                num_cls_fcs=1,
                num_mask_fcs=3,
                num_pos_fcs=2,
                num_center_fcs=2,
                num_offset_fcs=2,
                off_candidate_thr=0.6,
                act_cfg=dict(type='ReLU', inplace=True),
                center_type='binary',
                decoder_self_posembeds=dict(
                    type='PointConvBNPositionalEncoding',
                    input_channel=3,
                    num_pos_feats=128),
                pred_semantic=False,
                pred_coor_feat_mask=False,
                use_class_prior=False,
                last_layer=False,
                temp=1,
                ):
        super(PredictLayer, self).__init__()
        self.num_classes = num_classes
        self.off_candidate_thr = off_candidate_thr
        self.out_channels = out_channels
        self.center_type = center_type
        self.pred_semantic = pred_semantic
        self.pred_coor_feat_mask = pred_coor_feat_mask
        self.use_class_prior = use_class_prior
        self.last_layer = last_layer
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = self.num_classes - self.num_things_classes - 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temp = temp

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(build_activation_layer(act_cfg))
        self.fc_cls = nn.Linear(in_channels, self.num_classes)

        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.mask_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.mask_fcs.append(build_activation_layer(act_cfg))

        if self.use_class_prior and self.last_layer:
            self.cls_prior = nn.Linear(self.num_classes, in_channels*out_channels)
        else:
            self.fc_mask = nn.Linear(in_channels, out_channels)

        if self.pred_coor_feat_mask:
            self.coor_mask_fcs = nn.ModuleList()
            for _ in range(num_mask_fcs):
                self.coor_mask_fcs.append(
                    nn.Linear(in_channels, in_channels, bias=False))
                self.coor_mask_fcs.append(
                    build_norm_layer(dict(type='LN'), in_channels)[1])
                self.coor_mask_fcs.append(build_activation_layer(act_cfg))
            self.fc_coor_mask = nn.Linear(in_channels, out_channels)

        if self.pred_semantic:
            self.sem_fcs = nn.ModuleList()
            for _ in range(num_mask_fcs):
                self.sem_fcs.append(
                    nn.Linear(in_channels, in_channels, bias=False))
                self.sem_fcs.append(
                    build_norm_layer(dict(type='LN'), in_channels)[1])
                self.sem_fcs.append(build_activation_layer(act_cfg))
            self.fc_sem = nn.Linear(in_channels, out_channels)

    def forward_single(self, feature, indice, queries, feature_wo_indice=None, coor_feature=None):

        cls_feat = queries
        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fc_cls(cls_feat) #[N,cls_num]

        mask_feat = queries
        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)

        if self.use_class_prior and self.last_layer:
            cls_weight = cls_score.detach() * self.temp

            # for things predictions, only remain things weights.
            cls_weight[:, 0] = -10000
            cls_weight[:, self.num_stuff_classes:] = -10000

            softmax_cls_weight = cls_weight.softmax(-1)

            # for stuff predictions, directly assign one-hot weight.
            stuff_cls_weight = cls_weight.new_zeros([self.num_stuff_classes, self.num_classes])
            stuff_cls_weight[:,-self.num_stuff_classes:] = torch.eye(self.num_stuff_classes).to(cls_weight.device)

            softmax_cls_weight[-self.num_stuff_classes:] = stuff_cls_weight

            cls_fc_mask = self.cls_prior(softmax_cls_weight).view(cls_score.shape[0], self.in_channels, self.out_channels)
            mask_feat = torch.einsum('nb,nbd->nd', mask_feat, cls_fc_mask)
        else:
            mask_feat = self.fc_mask(mask_feat)

        mask_pred_fea = torch.einsum("nc,vc->vn", mask_feat, feature)

        if self.pred_coor_feat_mask:
            coor_mask_feat = queries
            for reg_layer in self.coor_mask_fcs:
                coor_mask_feat = reg_layer(coor_mask_feat)
            coor_mask_feat = self.fc_coor_mask(coor_mask_feat)
            coor_mask_pred = torch.einsum("nc,vc->vn", coor_mask_feat, coor_feature) 
            mask_pred = mask_pred_fea + coor_mask_pred
        else:
            coor_mask_pred = None
            mask_pred = mask_pred_fea

        sigmoid_masks = mask_pred.sigmoid()


                      
        if self.pred_semantic:
            sem_feat = queries
            for sem_layer in self.sem_fcs:
                sem_feat = sem_layer(sem_feat)
            sem_feat = self.fc_sem(sem_feat)
            sem_pred = torch.einsum("nc,vc->vn", sem_feat, feature) 
        else:
            sem_pred = None

        center = None
        # import pdb;pdb.set_trace()
        return cls_score, mask_pred.permute(1,0), sigmoid_masks, sem_pred, coor_mask_pred, mask_pred_fea


    def forward(self, feature_split, indice_split, queries,featurewoindice_split=None, coor_feature_list=None):
        if featurewoindice_split is None:
            featurewoindice_split = [None]*len(featurewoindice_split)
        if coor_feature_list is None or len(coor_feature_list) == 0:
            coor_feature_list = [None] * len(featurewoindice_split)

        cls_feat_list, mask_feat_list, sigmoid_masks_list, sigmoid_sem_pred, coor_mask_pred, mask_pred_fea = multi_apply(
                                                self.forward_single,
                                                feature_split,
                                                indice_split,
                                                queries,
                                                featurewoindice_split,
                                                coor_feature_list
                                                )            
        pred_dict = dict()
        pred_dict['labels'] = cls_feat_list
        pred_dict['masks'] = mask_feat_list
        pred_dict['sigmoid_masks'] = sigmoid_masks_list
        pred_dict['sem_masks'] = sigmoid_sem_pred
        pred_dict['coor_mask_pred'] = coor_mask_pred
        pred_dict['mask_pred_fea'] = mask_pred_fea

        return pred_dict

@TRANSFORMER_LAYER.register_module()
class UpdatorLayer(nn.Module):
    def __init__(self,
        updator_cfg=dict(
            type='DynamicConv',
            in_channels=128,
            feat_channels=64,
            out_channels=128,
            with_proj=False,
            act_cfg=dict(type='GeLU', inplace=True),
            norm_cfg=dict(type='LN',),
            ),
        in_channels=128,
        conv_kernel_size=1,
        num_heads=8,
        with_ffn=True,
        feedforward_channels=2048,
        num_ffn_fcs=2,
        dropout=0.0,
        ffn_act_cfg=dict(type='GeLU', inplace=True),
        decoder_self_posembeds=dict(
            type='PointConvBNPositionalEncoding',
            input_channel=3,
            num_pos_feats=128),
        decoder_cross_posembeds=dict(
            type='PointConvBNPositionalEncoding',
            input_channel=3,
            num_pos_feats=128),
        weigth_pos_posembeds=dict(
            type='PointConvBNPositionalEncoding',
            input_channel=3,
            num_pos_feats=32),
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.01),
        with_fea_fc=False,
        normalize_feat=False,
        feature_wo_indice=False,
        sigmoid_weight=False,):

        super(UpdatorLayer, self).__init__()

        self.with_ffn = with_ffn
        self.in_channels = in_channels
        self.pos_dim = in_channels
        self.feature_wo_indice = feature_wo_indice

        self.attention = MultiheadAttention(self.in_channels * conv_kernel_size,
                                            num_heads, dropout)
        self.attention_norm = build_norm_layer(
                                dict(type='LN'), self.in_channels * conv_kernel_size)[1]
        if self.with_ffn:
            self.ffn = FFN(self.in_channels,
                    feedforward_channels,
                    num_ffn_fcs,
                    act_cfg=ffn_act_cfg,
                    ffn_drop=dropout)
            self.ffn_norm = build_norm_layer(dict(type='LN'), self.in_channels)[1]

        self.with_fea_fc = with_fea_fc
        self.normalize_feat = normalize_feat
        self.sigmoid_weight = sigmoid_weight
        
        updator_cfg = copy.deepcopy(updator_cfg)

        self.kernel_update_conv = build_transformer(updator_cfg)

        self.attention = MultiheadAttention(self.in_channels * conv_kernel_size,
                                            num_heads, dropout)
        self.attention_norm = build_norm_layer(
                                dict(type='LN'), self.in_channels * conv_kernel_size)[1]
        if self.with_ffn:
            self.ffn = FFN(self.in_channels,
                    feedforward_channels,
                    num_ffn_fcs,
                    act_cfg=ffn_act_cfg,
                    ffn_drop=dropout)
            self.ffn_norm = build_norm_layer(dict(type='LN'), self.in_channels)[1]

    def forward_single(self, sigmoid_masks, feature, queries, indice, feature_wo_indice=None):

        if self.feature_wo_indice:
            feature = feature_wo_indice

        if self.sigmoid_weight:
            binary_masks = sigmoid_masks
        else:
            binary_masks = (sigmoid_masks>0.5).float()

        if self.normalize_feat:
            x_feat = torch.einsum('vn,vc->nc', binary_masks, feature)/((binary_masks.detach().sum(0)+1e-3)[:,None]) #[N,C]
        else:
            x_feat = torch.einsum('vn,vc->nc', binary_masks, feature)
        

        # import pdb;pdb.set_trace()
        queries = self.kernel_update_conv(queries, x_feat) 
        queries = self.attention_norm(self.attention(queries)) #[N,1,C]

        # [N,1,C] -> [1,N,C]
        queries = queries.permute(1,0,2)
        # FFN
        if self.with_ffn:
            queries = self.ffn_norm(self.ffn(queries))
        queries = queries.squeeze(0) #[N,C]

        return queries

    def forward(self, sigmoid_masks_list, feature_split_list, queries, indice_split_list=None, featurewoindice_split=None):
        new_queries = []
        if indice_split_list is None:
            indice_split_list = [None]*len(feature_split_list)
        if featurewoindice_split is None:
            featurewoindice_split = [None]*len(feature_split_list)

        for i in range(len(sigmoid_masks_list)):
            new_queries.append(self.forward_single(sigmoid_masks_list[i],
                                                                feature_split_list[i],
                                                                queries[i],
                                                                indice_split_list[i],
                                                                featurewoindice_split[i]))
        return new_queries

@TRANSFORMER_LAYER.register_module() 
class OriginAssignLayer(nn.Module):
    def __init__(self,
                assigner,
                sampler,
                num_classes,
                num_stuff_classes=None, 
                num_thing_classes=None,
                pos_weight=1.0,
                ):
        super().__init__()
        self.assigner = build_assigner(assigner)
        self.sampler = build_sampler(sampler)
        self.num_classes = num_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_thing_classes = num_thing_classes
        self.pos_weight = pos_weight
    
    def forward(self,pred_dict,target_dict,layer=None): 
        bs=len(pred_dict['masks'])

        sampling_results = []
        for b in range(bs):
            if layer == 0:
                assign_result = self.assigner.assign(pred_dict['masks'][b].detach(),None,
                            target_dict['masks'][b],target_dict['labels'][b])
            else:
                assign_result = self.assigner.assign(pred_dict['masks'][b].detach(),
                                        pred_dict['labels'][b].detach(),
                                        target_dict['masks'][b], target_dict['labels'][b]
                                        )
            sampler_dict = dict()
            sampler_dict['masks'] = target_dict['masks'][b]
            sampling_result = self.sampler.sample(assign_result,
                                                sampler_dict)
            sampling_results.append(sampling_result)

        target_dict_assign, weight_dict_assign = self.get_targets(sampling_results, bs=bs, pos_weight=self.pos_weight)

        return pred_dict, target_dict_assign, weight_dict_assign

    def get_targets(self,
                    sampling_results,
                    gt_sem_seg=None,
                    gt_sem_cls=None,
                    pos_weight=1.0,
                    bs=None,
                    ):
        if gt_sem_seg is None:
            gt_sem_seg = [None] * bs
            gt_sem_cls = [None] * bs

        target_dict_assign_list, weight_dict_assign_list = multi_apply(
            self._get_target_single,
            sampling_results,
            gt_sem_seg,
            gt_sem_cls,
            pos_weight=pos_weight)
        target_dict_assign = dict()
        for k in target_dict_assign_list[0].keys():
            target_dict_assign[k] = [d[k] for d in target_dict_assign_list]
        weight_dict_assign = dict()
        for k in weight_dict_assign_list[0].keys():
            weight_dict_assign[k] = [d[k] for d in weight_dict_assign_list]

        return target_dict_assign, weight_dict_assign

    def _get_target_single(self, sampling_result,
                           gt_sem_seg, gt_sem_cls, pos_weight,):

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_gt_masks = sampling_result.pos_gt_masks
        pos_gt_labels = sampling_result.pos_gt_labels   

        num_pos = pos_inds.shape[0]
        num_neg = neg_inds.shape[0]
        num_samples = num_pos + num_neg
        PTNUM = pos_gt_masks.shape[-1]
        labels = pos_gt_masks.new_full((num_samples, ),
                                   self.num_classes,
                                   dtype=torch.long)
        label_weights = pos_gt_masks.new_zeros(num_samples, self.num_classes)
        mask_targets = pos_gt_masks.new_zeros(num_samples, PTNUM)
        mask_weights = pos_gt_masks.new_zeros(num_samples, PTNUM)

        if num_pos > 0:
            pos_weight = 1.0 if pos_weight <= 0 else pos_weight

            labels[pos_inds] = pos_gt_labels
            label_weights[pos_inds] = pos_weight
            mask_targets[pos_inds, ...] = pos_gt_masks
            mask_weights[pos_inds, ...] = pos_weight

        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        if gt_sem_cls is not None and gt_sem_seg is not None:
            sem_labels = pos_gt_masks.new_full((self.num_stuff_classes, ),
                                           self.num_classes,
                                           dtype=torch.long)
            sem_targets = pos_gt_masks.new_zeros(self.num_stuff_classes, PTNUM)
            sem_weights = pos_gt_masks.new_zeros(self.num_stuff_classes, PTNUM)
            sem_stuff_weights = torch.eye(
                self.num_stuff_classes, device=pos_gt_masks.device)
            sem_thing_weights = pos_gt_masks.new_zeros(
                (self.num_stuff_classes, self.num_thing_classes))
            sem_label_weights = torch.cat(
                [sem_thing_weights, sem_stuff_weights], dim=-1)
                
            if len(gt_sem_cls > 0):
                sem_inds = gt_sem_cls - self.num_thing_classes
                sem_inds = sem_inds.long()
                sem_labels[sem_inds] = gt_sem_cls.long()
                sem_targets[sem_inds] = gt_sem_seg
                sem_weights[sem_inds] = 1
            
            label_weights[:, self.num_thing_classes:] = 0
            label_weights[:, 0] = 0
            labels = torch.cat([labels, sem_labels])
            label_weights = torch.cat([label_weights, sem_label_weights])
            mask_targets = torch.cat([mask_targets, sem_targets])
            mask_weights = torch.cat([mask_weights, sem_weights])
                
        target_dict_assign = dict()
        target_dict_assign['labels'] = labels
        target_dict_assign['masks'] = mask_targets

        if hasattr(sampling_result, 'pos_gt_offsets'):
            target_dict_assign['offsets'] = offset_targets
        if hasattr(sampling_result, 'pos_gt_centers'):
            target_dict_assign['centers'] = center_targets
        
        weight_dict_assign = dict()
        weight_dict_assign['labels'] = label_weights
        weight_dict_assign['masks'] = mask_weights

        return target_dict_assign, weight_dict_assign

@TRANSFORMER_LAYER.register_module() 
class SplitPrevAssignLayer(OriginAssignLayer):
    def __init__(self,
                assigner,
                sampler,
                num_classes,
                num_stuff_classes,
                num_thing_classes,
                ):
        super().__init__(assigner,
                        sampler,
                        num_classes,
                        num_stuff_classes,
                        num_thing_classes)
    
    def forward(self,pred_dict,target_dict,prev_pred_dict=None,layer=None):
        if prev_pred_dict is None:
            prev_pred_dict = pred_dict
        bs=len(pred_dict['masks'])
        sampling_results = []
        for b in range(bs):
            pred_mask_thing = prev_pred_dict['masks'][b][:-self.num_stuff_classes,:]#
            pred_label_thing = prev_pred_dict['labels'][b][:-self.num_stuff_classes,:]
            target_mask_thing = target_dict['thing_masks'][b]
            target_label_thing = target_dict['thing_labels'][b]

            if layer == 0:
                assign_result = self.assigner.assign(pred_mask_thing.detach(),None,
                            target_mask_thing,target_label_thing)
            else:
                assign_result = self.assigner.assign(pred_mask_thing.detach(),
                                        pred_label_thing.detach(),
                                        target_mask_thing, target_label_thing
                                        )
            sampler_dict = dict()
            sampler_dict['masks'] = target_mask_thing
            sampling_result = self.sampler.sample(assign_result,
                                                sampler_dict)
            sampling_results.append(sampling_result)

        target_dict_assign, weight_dict_assign = self.get_targets(sampling_results,
                                                                target_dict['stuff_masks'],
                                                                target_dict['stuff_labels'])


        if 'sparse_sem_labels' in target_dict:
            target_dict_assign['sparse_sem_labels'] = target_dict['sparse_sem_labels']
        return pred_dict, target_dict_assign, weight_dict_assign
   
@TRANSFORMER_LAYER.register_module()     
class LossLayer(nn.Module):
    def __init__(self,
                num_classes,
                num_thing_classes=9, 
                loss_cls=None,
                loss_cls2=None,
                loss_cls3=None,
                loss_clsmask=None,
                loss_mask=None,
                loss_dice=None,
                loss_center=None,
                loss_offset=None,
                loss_point_offset=None,
                loss_var=False,
                loss_var_weight=1,
                loss_sem=None,
                loss_sem_important=None,
                show_center=False,
                show_offset=False,
                show_point_offset=False,
                loss_rank=None,
                ignore_label=0,
                rank_type='all',
                cls_0layer=False,
                all_mask=False,
                loss_coor_dice_weight=1,
                loss_fea=False,
                layer_loss_weight=False,
                dont_pred_coor_layer0=False,
                ):
        super().__init__()
        # loss
        self.num_classes = num_classes
        self.num_thing_classes = num_thing_classes
        self.show_center = show_center
        self.show_offset = show_offset
        self.show_point_offset = show_point_offset
        self.rank_type = rank_type
        self.loss_var = loss_var
        self.loss_var_weight = loss_var_weight
        self.loss_sem_important = loss_sem_important
        self.loss_coor_dice_weight = loss_coor_dice_weight
        self.loss_fea = loss_fea
        self.layer_loss_weight = layer_loss_weight
        self.dont_pred_coor_layer0 = dont_pred_coor_layer0

        self.cls_0layer = cls_0layer
        self.all_mask = all_mask

        if loss_cls is not None:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls = None

        if loss_cls2 is not None:
            self.loss_cls2 = loss_cls2
        else:
            self.loss_cls2 = None
        
        if loss_cls3 is not None:
            self.loss_cls3 = build_loss(loss_cls3)
        else:
            self.loss_cls3 = None

        if loss_clsmask is not None:
            self.loss_clsmask = build_loss(loss_clsmask)
        else:
            self.loss_clsmask = None

        if loss_mask is not None:
            self.loss_mask = build_loss(loss_mask)
            if loss_mask['type'] == 'CrossEntropyLoss':
                self.maskce=True
            else:
                self.maskce=False
        else:
            self.loss_mask = None
        if loss_dice is not None:
            self.loss_dice = build_loss(loss_dice)
        else:
            self.loss_dice = None    


        self.loss_sem = loss_sem
        if loss_sem:
            self.loss_ce = nn.CrossEntropyLoss(ignore_index=0)
            self.loss_lovasz = build_loss(dict(
                                        type='Lovasz_Softmax',))
               
        self.ignore_label = ignore_label
    
    def forward(self,pred_dict,target_dict,weight_dict,layer,reduction_override=None,data_batch=None,zero_layer_pred=None):#TODO not implement weights yet
        
        bs = len(target_dict['labels'])
        label_targets = torch.cat(target_dict['labels'],0) # [bs*proposal_num]
        mask_targets = target_dict['masks'] # [bs(list),proposal_num,pt_num]
        pred_label = torch.cat(pred_dict['labels'],0) # [bs*proposal_num]
        pred_mask = pred_dict['masks'] # [bs(list),proposal_num,pt_num]
        label_weight = torch.cat(weight_dict['labels'],0) # [bs*proposal_num]
        mask_weight = weight_dict['masks'] # [bs(list),proposal_num,pt_num]
        pred_coor_mask = pred_dict['coor_mask_pred']
        mask_pred_fea = pred_dict['mask_pred_fea']

        if self.loss_sem and layer==0:
            pred_sem = pred_dict['sem_pred']
            sem_targets = target_dict['sparse_sem_labels']

        pos_inds = (label_targets > 0) & (label_targets < self.num_classes)
        bool_pos_inds = pos_inds.type(torch.bool)
        bool_pos_inds_split = bool_pos_inds.reshape(bs, -1)

        pos_thing_inds = (label_targets > 0) & (label_targets < self.num_thing_classes)
        bool_pos_thing_inds = pos_thing_inds.type(torch.bool)
        bool_pos_thing_inds_split = bool_pos_thing_inds.reshape(bs, -1)



        losses = dict()

        if self.loss_cls is not None:

            if layer == 0 and self.cls_0layer==False:
                losses[f'loss_cls_{layer}'] = pred_label.sum() * 0.0

                losses[f'pos_acc_{layer}'] = pred_label.sum() * 0.0
            else:
                num_pos = pos_inds.sum().float()
                avg_factor = reduce_mean(num_pos)

                losses[f'loss_cls_{layer}'] = self.loss_cls( 
                            pred_label,
                            label_targets) #TODO check the usage 

                losses[f'pos_acc_{layer}'] = accuracy(
                    pred_label[pos_inds], label_targets[pos_inds])

        if self.loss_mask is not None:
            loss_mask = 0
            valid_bs = 0
            if self.all_mask is True:
                for mask_idx,(mpred,mtarget) in enumerate(zip(pred_mask,mask_targets)):
                    # mp [pt_num,proposal_num]
                    # mt [proposal_num,pt_num]
                    # mp = mpred[bool_pos_inds_split[mask_idx]]
                    # mt = mtarget[bool_pos_inds_split[mask_idx]]
                    valid_bs += 1
                    if self.maskce:
                        loss_mask += self.loss_mask(mpred.reshape(-1,1),
                                                (mtarget).long().reshape(-1))
                    else:
                        loss_mask += self.loss_mask(mpred.reshape(-1,1),
                                                (1-mtarget).long().reshape(-1)) #避免focal loss expand one-hot 带来的问题 #1-!                

                if valid_bs > 0:
                    losses[f'loss_mask_{layer}'] = loss_mask / valid_bs
                else:
                    losses[f'loss_mask_{layer}'] = pred_label.sum() * 0.0 

            else:
                for mask_idx,(mpred,mtarget) in enumerate(zip(pred_mask,mask_targets)):
                    # mp [pt_num,proposal_num]
                    # mt [proposal_num,pt_num]
                    mp = mpred[bool_pos_inds_split[mask_idx]]
                    mt = mtarget[bool_pos_inds_split[mask_idx]]
                    if len(mp)>0:
                        valid_bs += 1
                    if self.maskce:
                        loss_mask += self.loss_mask(mpred.reshape(-1,1),
                                                (mtarget).long().reshape(-1))
                    else:
                        loss_mask += self.loss_mask(mpred.reshape(-1,1),
                                                (1-mtarget).long().reshape(-1)) #避免focal loss expand one-hot 带来的问题 #1-!                 

            if valid_bs > 0:
                losses[f'loss_mask_{layer}'] = loss_mask / valid_bs
            else:
                losses[f'loss_mask_{layer}'] = pred_label.sum() * 0.0 
        
        if self.loss_dice is not None:
            loss_dice = 0
            valid_bs=0
            if self.all_mask is True:
                for mask_idx,(mpred,mtarget) in enumerate(zip(pred_mask,mask_targets)):
                    # mp [pt_num,proposal_num]
                    # mt [proposal_num,pt_num]
                    # mp = mpred[bool_pos_inds_split[mask_idx]]
                    # mt = mtarget[bool_pos_inds_split[mask_idx]]
                    valid_bs += 1
                    loss_dice += self.loss_dice(mpred, mtarget)             

                if valid_bs > 0:
                    losses[f'loss_dice_{layer}'] = loss_dice / valid_bs
                else:
                    losses[f'loss_dice_{layer}'] = pred_label.sum() * 0.0
            else: 
                for mask_idx,(mpred,mtarget) in enumerate(zip(pred_mask,mask_targets)):
                    # mp [pt_num,proposal_num]
                    # mt [proposal_num,pt_num]
                    mp = mpred[bool_pos_inds_split[mask_idx]]
                    mt = mtarget[bool_pos_inds_split[mask_idx]]
                    if len(mp)>0:
                        valid_bs += 1               
                        loss_dice += self.loss_dice(mp, mt)

                if valid_bs > 0:
                    losses[f'loss_dice_{layer}'] = loss_dice / valid_bs # dropout　会导致dice不下降?
                else:
                    losses[f'loss_dice_{layer}'] = pred_label.sum() * 0.0

        if self.loss_dice is not None and pred_coor_mask[0] is not None:
            if self.dont_pred_coor_layer0 and layer == 0:
                pass
            else:
                if self.layer_loss_weight:
                    loss_coor_dice_weight = self.loss_coor_dice_weight * (layer+1)
                else:
                    loss_coor_dice_weight = self.loss_coor_dice_weight

                loss_coor_dice = 0
                valid_bs = 0

                for mask_idx, (mpred, mtarget) in enumerate(
                        zip(pred_coor_mask, mask_targets)):
                    # mp [pt_num,proposal_num]
                    # mt [proposal_num,pt_num]
                    mpred = mpred.permute(1,0)[:128]
                    mtarget = mtarget[:128]
                    mp = mpred[bool_pos_inds_split[mask_idx][:128]]
                    mt = mtarget[bool_pos_inds_split[mask_idx][:128]]
                    if len(mp) > 0:
                        valid_bs += 1
                        loss_coor_dice += self.loss_dice(mp, mt)
                if valid_bs > 0:
                    losses[
                        f'loss_coor_dice_{layer}'] = loss_coor_dice / valid_bs * loss_coor_dice_weight 
                else:
                    losses[f'loss_coor_dice_{layer}'] = pred_label[0].sum() * 0.0

        if self.loss_sem is not None and layer==0:
            loss_lovasz = 0
            loss_ce = 0
            for  sp,st in zip(pred_sem,sem_targets):
                loss_lovasz += self.loss_lovasz(nn.functional.softmax(sp), st)
                loss_ce += self.loss_ce(sp, st)
            
            losses['loss_lovasz'] = loss_lovasz/len(pred_sem)
            losses['loss_ce'] = loss_ce/len(pred_sem)
        
        return losses 

@TRANSFORMER_LAYER.register_module()     
class GtConvertLayer():
    def __init__(self, thing_class, task='semantic',minpoint=0,point_wise=False, point_cloud_range=None):
        super().__init__()
        self.task=task
        self.minpoint=minpoint
        self.point_wise=point_wise
        self.thing_class=thing_class
        self.thingnum = len(thing_class)
        self.point_cloud_range = point_cloud_range

        if point_cloud_range[1]=='-np.pi':
            point_cloud_range[1]=-np.pi
            point_cloud_range[4]=np.pi

    def convert(self,data_batch):
        target_dict = dict()


        if self.task == 'semantic':
            labels_list, masks_list = self.sem2mask_voxel(data_batch)
            target_dict['labels'] = labels_list
            target_dict['masks'] = masks_list
        elif self.task == 'panoptic':
            labels_list, masks_list, tl,tm,sl,sm = self.semins2mask_voxel(data_batch)
            target_dict['labels'] = labels_list
            target_dict['masks'] = masks_list
            target_dict['thing_labels'] = tl
            target_dict['thing_masks'] = tm
            target_dict['stuff_labels'] = sl
            target_dict['stuff_masks'] = sm
        elif self.task == 'instance':
            labels_list, masks_list = self.ins2mask_voxel(data_batch)
            target_dict['labels'] = labels_list
            target_dict['masks'] = masks_list    
        else:
            raise NotImplementedError

        return target_dict

    def semins2mask_voxel(self, data_batch):
        labels_list=[]
        masks_list=[]

        thing_labels_list=[]
        thing_masks_list=[]       
        stuff_labels_list=[]
        stuff_masks_list=[]

        for idx in range(len(data_batch['sparse_sem_labels'])):
            if self.point_wise:
                gt_sem_seg = data_batch['pt_labs'][idx]
                gt_inst_seg = data_batch['sort_inst_labels'][idx]

            else:
                gt_sem_seg = data_batch['sparse_sem_labels'][idx]
                gt_inst_seg = data_batch['sparse_inst_labels'][idx]

            valid_instance = (gt_sem_seg>0) & (gt_sem_seg<self.thingnum)

            gt_inst_seg = (gt_inst_seg << 16) + gt_sem_seg
            sem_classes = torch.unique(gt_sem_seg)
            inst_classes = torch.unique(gt_inst_seg)

            masks = []
            labels = []
            thing_masks = []
            thing_labels = []
            stuff_masks = []
            stuff_labels = []

            for i in inst_classes:
                sem = i & 0xFFFF
                if sem==8:
                    a = 1
                if sem in self.thing_class:
                    if (gt_inst_seg == i).sum()>self.minpoint:
                        labels.append(sem)
                        tmpmask = gt_inst_seg == i
                        masks.append(tmpmask)

                        thing_labels.append(sem)
                        thing_masks.append(tmpmask) 

            for i in sem_classes: 
                if (i in self.thing_class) or (i == 0):
                    continue
                if (gt_sem_seg == i).sum()>self.minpoint:
                    labels.append(i)
                    masks.append(gt_sem_seg == i)
                    stuff_labels.append(i)
                    stuff_masks.append(gt_sem_seg == i) 
          
            if len(labels) > 0:
                labels = torch.stack(labels)
                masks = torch.stack(masks,dim=0)
            else:
                labels = gt_sem_seg.new_zeros(size=[0])
                masks = gt_sem_seg.new_zeros(
                    size=[0, gt_sem_seg.shape[-1]])

            if len(thing_labels) > 0:
                thing_labels = torch.stack(thing_labels)
                thing_masks = torch.stack(thing_masks,dim=0)
            else:
                thing_labels = gt_sem_seg.new_zeros(size=[0])
                thing_masks = gt_sem_seg.new_zeros(
                    size=[0, gt_sem_seg.shape[-1]])
            
            if len(stuff_labels) > 0:
                stuff_labels = torch.stack(stuff_labels)
                stuff_masks = torch.stack(stuff_masks,dim=0)
            else:
                stuff_labels = gt_sem_seg.new_zeros(size=[0])
                stuff_masks = gt_sem_seg.new_zeros(
                    size=[0, gt_sem_seg.shape[-1]])



            labels, masks = labels.long(), masks.float()
         

            thing_labels, thing_masks = thing_labels.long(), thing_masks.float()
            stuff_labels, stuff_masks = stuff_labels.long(), stuff_masks.float()

            labels_list.append(labels)
            masks_list.append(masks)

            thing_labels_list.append(thing_labels)
            thing_masks_list.append(thing_masks)     
            stuff_labels_list.append(stuff_labels)
            stuff_masks_list.append(stuff_masks)

        return labels_list, masks_list, thing_labels_list, thing_masks_list, \
                                            stuff_labels_list, stuff_masks_list

    def sem2mask_voxel(self, data_batch):
        labels_list=[]
        masks_list=[]
        for gt_sem_seg in data_batch['sparse_sem_labels']:
            sem_classes = torch.unique(gt_sem_seg)
            masks = []
            labels = []

            for i in sem_classes:
                labels.append(i)
                masks.append(gt_sem_seg == i)
            
            if len(labels) > 0:
                labels = torch.stack(labels)
                masks = torch.stack(masks,dim=0)
            else:
                labels = gt_sem_seg.new_zeros(size=[0])
                masks = gt_sem_seg.new_zeros(
                    size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]])

            labels, masks = labels.long(), masks.float()
            labels_list.append(labels)
            masks_list.append(masks)

        return labels_list, masks_list

    def ins2mask_voxel(self, data_batch):
        labels_list=[]
        masks_list=[]
        for gt_sem_seg,gt_inst_seg in zip(data_batch['sparse_sem_labels'],data_batch['sparse_inst_labels']):
            gt_inst_seg = (gt_inst_seg << 16) + gt_sem_seg
            sem_classes = torch.unique(gt_sem_seg)
            inst_classes = torch.unique(gt_inst_seg)
            # classes ranges from 0 - N-1, where the class IDs in
            # [0, num_thing_classes - 1] are IDs of thing classes
            masks = []
            labels = []
            for i in inst_classes:
                sem = i & 0xFFFF
                if sem in self.thing_class:
                    labels.append(sem)
                    masks.append(gt_inst_seg == i)

            for i in sem_classes:
                continue
            
            if len(labels) > 0:
                labels = torch.stack(labels)
                #masks = torch.cat(masks)
                masks = torch.stack(masks,dim=0)
            else:
                labels = gt_sem_seg.new_zeros(size=[0])
                masks = gt_sem_seg.new_zeros(
                    size=[0, gt_sem_seg.shape[-1]])

            labels, masks = labels.long(), masks.float()
            labels_list.append(labels)
            masks_list.append(masks)

        return labels_list, masks_list

@TRANSFORMER_LAYER.register_module()
class NewGetPanopticLayer():
    def __init__(self,
                num_thing_classes,
                num_proposals,
                num_stuff_classes,
                ignore_class,
                classnum,
                thing_class,
                object_mask_thr=0.4,
                iou_thr=0.8,
                small_obj=[2,3,4,5,6,7,8]):
        self.object_mask_thr = object_mask_thr
        self.iou_thr = iou_thr
        self.num_thing_classes = num_thing_classes
        self.num_proposals = num_proposals
        self.num_stuff_classes = num_stuff_classes
        self.num_thing_proposals = self.num_proposals - num_stuff_classes
        self.ignore_class = ignore_class
        self.classnum = classnum
        self.thing_class = thing_class
        self.small_obj = small_obj
        self.small_obj = [None] # don't use it now

    def get_panoptic(self,pred_logits_list, pred_masks_list, points=None, pred_sem=None, gt=None, pred_dict_list=None, target_dict=None):#iou_thr变大,IOU必定变小,RQ可能变大

        pt_sem_preds_list = []
        ins_id_list = []
        for i in range(len(pred_logits_list)):
            pred_logits = pred_logits_list[i]
            pred_masks = pred_masks_list[i]

            scores = pred_logits[:self.num_thing_proposals][:, 1:self.num_thing_classes] #排除0
            thing_scores, thing_labels = scores.sigmoid().max(dim=1)
            thing_labels = thing_labels + 1 
            stuff_scores = pred_logits[
                self.num_thing_proposals:][:, self.num_thing_classes:].diag().sigmoid()
            stuff_labels = torch.arange(
                0, self.num_stuff_classes) + self.num_thing_classes
            stuff_labels = stuff_labels.to(thing_labels.device)


            scores = torch.cat([thing_scores*2, stuff_scores], dim=0)
            labels = torch.cat([thing_labels, stuff_labels], dim=0)


            small_obj = self.small_obj
            keep = ((scores > self.object_mask_thr) & (labels != 0)).bool()

            cur_scores = scores[keep] # [pos_proposal_num]
            cur_classes = labels[keep] # [pos_proposal_num]
            cur_masks = pred_masks[keep] # [pos_proposal_num, pt_num]
            cur_masks = cur_masks.sigmoid()

            if pred_sem is not None:
                true_label = []
                for iss in range(len(cur_masks)):
                    m = cur_masks[iss]>0.5
                    if m.sum()==0:
                        true_label.append(m.new_zeros([]).long())
                    else:
                        unique, counts = torch.unique(pred_sem[i][m], return_counts=True)
                        idx = torch.argmax(counts)
                        if unique[idx]==0:
                            counts[idx]=0
                            idx = torch.argmax(counts,-1)
                        # if unique[idx]==0:
                        #     sss = 1
                        true_label.append(unique[idx])
                true_label = torch.stack(true_label,0)
                cur_classes = true_label

            pt_sem_preds = cur_classes.new_full((cur_masks.shape[-1],),self.ignore_class[0])
            ins_id = cur_classes.new_full((cur_masks.shape[-1],),self.ignore_class[0])

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                pt_sem_preds_list.append(pt_sem_preds.cpu())
                ins_id_list.append(ins_id.cpu())
                continue


            cur_prob_masks = cur_masks * cur_scores.reshape(-1,1)
            cur_mask_ids = cur_prob_masks.argmax(0)
            instance_id = 1

            # idx = torch.linspace(0,127,128)
            # print(idx[keep][cur_classes==9])

            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class in self.thing_class
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                if mask_area > 0 and original_area > 0: 
                    if mask_area / original_area < self.iou_thr:#TODO weird phenomena 
                        continue

                    pt_sem_preds[mask] = pred_class 
                    
                    if isthing:
                        
                        ins_id[mask] = instance_id
                        instance_id += 1
            pt_sem_preds_list.append(pt_sem_preds.cpu())
            ins_id_list.append(ins_id.cpu())

        return pt_sem_preds_list, ins_id_list

    def get_panoptic_ensemble(self,
                     pred_logits_list,
                     pred_masks_list,
                     ):

        pt_sem_preds_list = []
        ins_id_list = []
        for i in range(len(pred_logits_list)):
            pred_logits = pred_logits_list[i]
            pred_masks = pred_masks_list[i].sigmoid()          

            self.num_proposals = pred_logits.shape[0]//2
            self.num_thing_proposals = self.num_proposals - self.num_stuff_classes

            if self.num_thing_proposals > 0:
                thing_idx = np.concatenate([np.arange(0, self.num_thing_proposals)])#, np.arange(self.num_proposals, self.num_proposals+self.num_thing_proposals)])
                stuff_idx1 = np.arange(self.num_thing_proposals, self.num_proposals)
                stuff_idx2 = np.arange(self.num_proposals+self.num_thing_proposals, self.num_proposals*2)
                scores = pred_logits[thing_idx][:, 1:self.num_thing_classes].sigmoid()
                thing_scores, thing_labels = scores.max(dim=1)
                thing_labels = thing_labels + 1
                stuff_scores = torch.cat([pred_logits[stuff_idx1][:,
                                               self.num_thing_classes:].diag(
                                               ).sigmoid(),
                                               pred_logits[stuff_idx2][:,
                                               self.num_thing_classes:].diag(
                                               ).sigmoid()])
                stuff_labels = torch.arange(
                    0, self.num_stuff_classes) + self.num_thing_classes
                stuff_labels = torch.cat([stuff_labels,stuff_labels]).to(thing_labels.device)

                scores = torch.cat([thing_scores*2, stuff_scores], dim=0)
                labels = torch.cat([thing_labels, stuff_labels], dim=0)

                pred_masks = pred_masks[np.concatenate([thing_idx, stuff_idx1, stuff_idx2])]
            else:
                stuff_scores = pred_logits[
                    self.num_thing_proposals:][:,
                                               self.num_thing_classes:].diag(
                                               ).sigmoid()
                stuff_labels = torch.arange(
                    0, self.num_stuff_classes) + self.num_thing_classes
                stuff_labels = stuff_labels.to(stuff_scores.device)

                scores = stuff_scores
                labels = stuff_labels
            keep = ((scores > self.object_mask_thr) & (labels != 0)).bool()
            cur_scores = scores[keep]  # [pos_proposal_num]
            cur_classes = labels[keep]  # [pos_proposal_num]
            cur_masks = pred_masks[keep]  # [pos_proposal_num, pt_num]


            binary_masks = (pred_masks>0.5).float()
            overlap = torch.einsum('nc,mc->nm', binary_masks, binary_masks)
            oversum = (binary_masks.sum(1)[None,:] + binary_masks.sum(1)[:,None]) - overlap
            iou = overlap / (oversum + 1e-8)

            #nms
            for iou_idx in range(iou.shape[0]):
                idx = torch.nonzero(iou[iou_idx]>0.5)
                for id in idx:
                    if id != iou_idx:
                        if scores[id]>scores[iou_idx]:
                            iou[iou_idx] = 0
                            continue
                        else:
                            iou[id] = 0
            keep = (iou.sum(1)>0).bool()
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]

            pt_sem_preds = cur_classes.new_full((cur_masks.shape[-1], ),
                                                self.ignore_class[0])
            ins_id = cur_classes.new_full((cur_masks.shape[-1], ),
                                          self.ignore_class[0])

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                pt_sem_preds_list.append(pt_sem_preds.cpu())
                ins_id_list.append(ins_id.cpu())
                continue

            cur_prob_masks = cur_masks * cur_scores.reshape(-1,
                                                            1)
            cur_mask_ids = cur_prob_masks.argmax(
                0) 
            instance_id = 1

            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class in self.thing_class
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                if mask_area > 0 and original_area > 0: 
                    if mask_area / original_area < self.iou_thr:
                        continue
                    pt_sem_preds[mask] = pred_class

                    if isthing:
                        ins_id[
                            mask] = instance_id  # << INSTANCE_OFFSET# + pred_class
                        instance_id += 1
            pt_sem_preds_list.append(pt_sem_preds.cpu())
            ins_id_list.append(ins_id.cpu())
        return pt_sem_preds_list, ins_id_list



