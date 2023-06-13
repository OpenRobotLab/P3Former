import torch
from mmdet3d.registry import MODELS
from mmdet3d.models.segmentors.cylinder3d import Cylinder3D
from mmdet3d.structures import PointData
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig

@MODELS.register_module()
class _P3Former(Cylinder3D):
    """P3Former."""

    def __init__(self,
                 voxel_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 loss_regularization: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(voxel_encoder=voxel_encoder,
                        backbone=backbone,
                        decode_head=decode_head,
                        neck=neck,
                        auxiliary_head=auxiliary_head,
                        loss_regularization=loss_regularization,
                        train_cfg=train_cfg,
                        test_cfg=test_cfg,
                        data_preprocessor=data_preprocessor,
                        init_cfg=init_cfg)

    def loss(self, batch_inputs_dict,batch_data_samples):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone
        x = self.extract_feat(batch_inputs_dict)
        batch_inputs_dict['features'] = x.features
        losses = dict()
        loss_decode = self._decode_head_forward_train(batch_inputs_dict, batch_data_samples)
        losses.update(loss_decode)

        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        x = self.extract_feat(batch_inputs_dict)
        batch_inputs_dict['features'] = x.features
        pts_semantic_preds, pts_instance_preds = self.decode_head.predict(batch_inputs_dict, batch_data_samples)
        return self.postprocess_result(pts_semantic_preds, pts_instance_preds, batch_data_samples)

    def postprocess_result(self, pts_semantic_preds, pts_instance_preds, batch_data_samples):
        for i in range(len(pts_semantic_preds)):
            batch_data_samples[i].set_data(
                {'pred_pts_seg': PointData(**{'pts_semantic_mask': pts_semantic_preds[i],
                                                'pts_instance_mask': pts_instance_preds[i]})})
        return batch_data_samples
