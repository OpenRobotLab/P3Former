import warnings
from mmcv.cnn import (ConvModule, build_activation_layer, build_norm_layer,
                      constant_init)
from mmcv.runner import BaseModule
from torch import nn as nn
import numpy as np
#from mmdet3d.ops import spconv as spconv
import torch

is_new_conv = True
try:
    import spconv.pytorch as spconv
except ImportError:
    import spconv
    is_new_conv = False

from mmdet.models import BACKBONES


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, norm_cfg, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = build_norm_layer(norm_cfg, out_filters)[1]

        self.init_weights()
    def init_weights(self):
        if self.bn0 is not None:
            constant_init(self.bn0, val=1, bias=0)
        if self.bn0_2 is not None:
            constant_init(self.bn0_2, val=1, bias=0)
        if self.bn1 is not None:
            constant_init(self.bn1, val=1, bias=0)
        if self.bn2 is not None:
            constant_init(self.bn2, val=1, bias=0)

    def forward(self, x):
        shortcut = self.conv1(x)
        #first rule than batchnorm
        if is_new_conv:
            shortcut = shortcut.replace_feature(self.act1(shortcut.features))
            shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
            shortcut = self.conv1_2(shortcut)
            shortcut.features = shortcut.replace_feature(self.act1_2(shortcut.features))
            shortcut.features = shortcut.replace_feature(self.bn0_2(shortcut.features))

            resA = self.conv2(x)
            resA.features = resA.replace_feature(self.act2(resA.features))
            resA.features = resA.replace_feature(self.bn1(resA.features))

            resA = self.conv3(resA)
            resA.features = resA.replace_feature(self.act3(resA.features))
            resA.features = resA.replace_feature(self.bn2(resA.features))
            resA.features = resA.replace_feature(resA.features + shortcut.features)

        else:
            shortcut.features = self.act1(shortcut.features)
            shortcut.features = self.bn0(shortcut.features)

            shortcut = self.conv1_2(shortcut)
            shortcut.features = self.act1_2(shortcut.features)
            shortcut.features = self.bn0_2(shortcut.features)

            resA = self.conv2(x)
            resA.features = self.act2(resA.features)
            resA.features = self.bn1(resA.features)

            resA = self.conv3(resA)
            resA.features = self.act3(resA.features)
            resA.features = self.bn2(resA.features)
            resA.features = resA.features + shortcut.features

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, norm_cfg, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn0 = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = build_norm_layer(norm_cfg, out_filters)[1]

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.init_weights()

    def init_weights(self):
        if self.bn0 is not None:
            constant_init(self.bn0, val=1, bias=0)
        if self.bn0_2 is not None:
            constant_init(self.bn0_2, val=1, bias=0)
        if self.bn1 is not None:
            constant_init(self.bn1, val=1, bias=0)
        if self.bn2 is not None:
            constant_init(self.bn2, val=1, bias=0)
        #self.weight_initialization()#TODO

    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x):

        if is_new_conv:
            shortcut = self.conv1(x)
            shortcut.features = shortcut.replace_feature(self.act1(shortcut.features))
            shortcut.features = shortcut.replace_feature(self.bn0(shortcut.features))

            shortcut = self.conv1_2(shortcut)
            shortcut.features = shortcut.replace_feature(self.act1_2(shortcut.features))
            shortcut.features = shortcut.replace_feature(self.bn0_2(shortcut.features))

            resA = self.conv2(x)
            resA.features = resA.replace_feature(self.act2(resA.features))
            resA.features = resA.replace_feature(self.bn1(resA.features))

            resA = self.conv3(resA)
            resA.features = resA.replace_feature(self.act3(resA.features))
            resA.features = resA.replace_feature(self.bn2(resA.features))

            resA.features = resA.replace_feature(resA.features + shortcut.features)

        else:
            shortcut = self.conv1(x)
            shortcut.features = self.act1(shortcut.features)
            shortcut.features = self.bn0(shortcut.features)

            shortcut = self.conv1_2(shortcut)
            shortcut.features = self.act1_2(shortcut.features)
            shortcut.features = self.bn0_2(shortcut.features)

            resA = self.conv2(x)
            resA.features = self.act2(resA.features)
            resA.features = self.bn1(resA.features)

            resA = self.conv3(resA)
            resA.features = self.act3(resA.features)
            resA.features = self.bn2(resA.features)

            resA.features = resA.features + shortcut.features

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, norm_cfg, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act1 = nn.LeakyReLU()
        self.bn1 = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key)
        self.act3 = nn.LeakyReLU()
        self.bn3 = build_norm_layer(norm_cfg, out_filters)[1]
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.init_weights()

    def init_weights(self):
        if self.bn1 is not None:
            constant_init(self.bn1, val=1, bias=0)
        if self.bn2 is not None:
            constant_init(self.bn2, val=1, bias=0)
        if self.bn3 is not None:
            constant_init(self.bn3, val=1, bias=0)

        # self.weight_initialization()#TODO

    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):

        if is_new_conv:
            upA = self.trans_dilao(x)
            upA.features = upA.replace_feature(self.trans_act(upA.features))
            upA.features = upA.replace_feature(self.trans_bn(upA.features))

            ## upsample
            upA = self.up_subm(upA)

            upA.features = upA.replace_feature(upA.features + skip.features)

            upE = self.conv1(upA)
            upE.features = upE.replace_feature(self.act1(upE.features))
            upE.features = upE.replace_feature(self.bn1(upE.features))

            upE = self.conv2(upE)
            upE.features = upE.replace_feature(self.act2(upE.features))
            upE.features = upE.replace_feature(self.bn2(upE.features))

            upE = self.conv3(upE)
            upE.features = upE.replace_feature(self.act3(upE.features))
            upE.features = upE.replace_feature(self.bn3(upE.features))
        
        else:
            upA = self.trans_dilao(x)
            upA.features = self.trans_act(upA.features)
            upA.features = self.trans_bn(upA.features)

            ## upsample
            upA = self.up_subm(upA)

            upA.features = upA.features + skip.features

            upE = self.conv1(upA)
            upE.features = self.act1(upE.features)
            upE.features = self.bn1(upE.features)

            upE = self.conv2(upE)
            upE.features = self.act2(upE.features)
            upE.features = self.bn2(upE.features)

            upE = self.conv3(upE)
            upE.features = self.act3(upE.features)
            upE.features = self.bn3(upE.features)

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, norm_cfg, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_3 = build_norm_layer(norm_cfg, out_filters)[1]#TODO check why source code why not init bn
        self.act1_3 = nn.Sigmoid()

    def init_weights(self):
        if self.bn0 is not None:
            constant_init(self.bn0, val=1, bias=0)
        if self.bn0_2 is not None:
            constant_init(self.bn0_2, val=1, bias=0)
        if self.bn0_3 is not None:
            constant_init(self.bn0_3, val=1, bias=0)

    def forward(self, x):
        if is_new_conv:
            shortcut = self.conv1(x)
            shortcut.features = shortcut.replace_feature(self.bn0(shortcut.features))
            shortcut.features = shortcut.replace_feature(self.act1(shortcut.features))

            shortcut2 = self.conv1_2(x)
            shortcut2.features = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
            shortcut2.features = shortcut2.replace_feature(self.act1_2(shortcut2.features))

            shortcut3 = self.conv1_3(x)
            shortcut3.features = shortcut3.replace_feature(self.bn0_3(shortcut3.features))
            shortcut3.features = shortcut3.replace_feature(self.act1_3(shortcut3.features))
            shortcut.features = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)

            shortcut.features = shortcut.replace_feature(shortcut.features * x.features)
        else:
            shortcut = self.conv1(x)
            shortcut.features = self.bn0(shortcut.features)
            shortcut.features = self.act1(shortcut.features)

            shortcut2 = self.conv1_2(x)
            shortcut2.features = self.bn0_2(shortcut2.features)
            shortcut2.features = self.act1_2(shortcut2.features)

            shortcut3 = self.conv1_3(x)
            shortcut3.features = self.bn0_3(shortcut3.features)
            shortcut3.features = self.act1_3(shortcut3.features)
            shortcut.features = shortcut.features + shortcut2.features + shortcut3.features

            shortcut.features = shortcut.features * x.features
            
        return shortcut


@BACKBONES.register_module()
class Asymm_3d_spconv(BaseModule):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 n_height=32, strict=False, init_size=16, norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super(Asymm_3d_spconv, self).__init__()
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        # print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre", norm_cfg=norm_cfg)
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2", norm_cfg=norm_cfg)
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3", norm_cfg=norm_cfg)
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4", norm_cfg=norm_cfg)
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5", norm_cfg=norm_cfg)

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5", norm_cfg=norm_cfg)
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4", norm_cfg=norm_cfg)
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3", norm_cfg=norm_cfg)
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2", norm_cfg=norm_cfg)

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon", norm_cfg=norm_cfg)
        


    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e.features = torch.cat((up0e.features, up1e.features), 1)

        
        return up0e, up0e
