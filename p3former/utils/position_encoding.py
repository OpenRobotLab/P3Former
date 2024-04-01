# Copyright (c) Facebook, Inc. and its affiliates.
# # Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
"""
Various positional encodings for the transformer.
"""
import math

import torch
from torch import nn
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.cnn import build_norm_layer

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

@POSITIONAL_ENCODING.register_module()
class PointConvBNPositionalEncoding(nn.Module):
    """Absolute position embedding with Conv learning.

    Args:
        input_channel (int): input features dim.
        num_pos_feats (int): output position features dim.
            Defaults to 288 to be consistent with seed features dim.
    """

    def __init__(self, input_channel, num_pos_feats=288, norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.01)):
        super().__init__()
        # self.position_embedding_head = nn.Sequential(
        #     nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
        #     nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True),
        #     nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            build_norm_layer(norm_cfg, num_pos_feats)[1],
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats))

    def forward(self, xyz):
        """Forward pass.

        Args:
            xyz (Tensor)： (N, 4) the coordinates to embed.

        Returns:
            Tensor: (B, num_pos_feats, N) the embeded position features.
        """
        #xyz = xyz.permute(0, 2, 1)
        position_embedding = self.position_embedding_head(xyz.float())
        return position_embedding


@POSITIONAL_ENCODING.register_module()
class PointSinePositionalEncoding(nn.Module):
    """Absolute position embedding with Conv learning.

    Args:
        input_channel (int): input features dim.
        num_pos_feats (int): output position features dim.
            Defaults to 288 to be consistent with seed features dim.
    """

    def __init__(self,  num_feats=64, 
                        temperature=10000,
                        normalize=False,
                        scale=2 * math.pi,
                        eps=1e-6,
                        offset=0.,
                        init_cfg=None):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, xy):
        y_embed = xy[:,0]
        x_embed = xy[:,1]
        if self.normalize:
            raise NotImplementedError
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32).cuda()
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        pos_x = torch.cat(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()),dim=1)
        pos_y = torch.cat(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()),dim=1)
        pos = torch.cat((pos_y, pos_x), dim=1)
        return pos