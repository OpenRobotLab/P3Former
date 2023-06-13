# Copyright (c) OpenMMLab. All rights reserved.
"""copy from
https://github.com/ZwwWayne/K-Net/blob/main/knet/det/mask_pseudo_sampler.py."""

import torch
from mmengine.structures import InstanceData

from mmdet3d.registry import TASK_UTILS
from mmdet.models.task_modules.assigners import AssignResult
from torch import Tensor
from mmdet.models.task_modules.samplers.base_sampler import BaseSampler
from mmdet.models.task_modules.samplers.mask_sampling_result import MaskSamplingResult


class _MaskSamplingResult(MaskSamplingResult):
    """Mask sampling result."""

    def __init__(self,
                 pos_inds: Tensor,
                 neg_inds: Tensor,
                 masks: Tensor,
                 gt_masks: Tensor,
                 assign_result: AssignResult,
                 gt_flags: Tensor,
                 avg_factor_with_neg: bool = True) -> None:
        super().__init__(pos_inds=pos_inds,
                         neg_inds=neg_inds,
                         masks=masks,
                         gt_masks=gt_masks,
                         assign_result=assign_result,
                         gt_flags=gt_flags,
                         avg_factor_with_neg=avg_factor_with_neg)
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None


@TASK_UTILS.register_module()
class _MaskPseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result: AssignResult, pred_instances: InstanceData,
               gt_instances: InstanceData, *args, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Mask assigning results.
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``scores`` and ``masks`` predicted
                by the model.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``labels`` and ``masks``
                attributes.

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pred_masks = pred_instances.masks
        gt_masks = gt_instances.masks
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = pred_masks.new_zeros(pred_masks.shape[0], dtype=torch.uint8)
        sampling_result = _MaskSamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            masks=pred_masks,
            gt_masks=gt_masks,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False)
        return sampling_result
