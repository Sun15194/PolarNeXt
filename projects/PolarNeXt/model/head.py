import copy
import logging
import math
import torch
import cv2
import numpy as np
import torch.nn as nn

from mmcv.ops import batched_nms
from mmcv.cnn import Scale, ConvModule
from mmengine import print_log
from mmengine.structures import InstanceData

from mmengine.config import ConfigDict
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from mmdet.models.utils import multi_apply, select_single_mlvl, filter_scores_and_topk
from mmdet.models import AnchorFreeHead

INF = 1e8


@MODELS.register_module()
class PolarNeXtHead(AnchorFreeHead):
    def __init__(self,
                 num_rays: int = 36,
                 num_sample: int = 9,
                 num_classes: int = 80,
                 in_channels: int = 256,
                 mask_size: Tuple = (64, 64),
                 align_offset: float = 0.5,
                 sampling_radius: float = 1.5,
                 regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),
                                              (256, 512), (512, INF)),
                 loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     loss_weight=1.0),
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.regress_ranges = regress_ranges
        self.num_rays = num_rays
        self.num_sample = num_sample
        self.mask_size = mask_size
        self.align_offset = align_offset
        self.sampling_radius = sampling_radius
        self.angles = torch.arange(0, 360, 360 / self.num_rays).cuda() / 180 * math.pi

        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(
            self.feat_channels, self.num_rays, 3, padding=1)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        return 1

    def get_targets(
            self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor]]:
        return 1, 1

    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        return multi_apply(self.forward_single, x, self.scales, self.strides)

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor, Tensor]:
        cls_score, poly_pred, _, reg_feat = super().forward_single(x)

        centerness = self.conv_centerness(reg_feat)

        poly_pred = scale(poly_pred).float()
        poly_pred *= stride
        poly_pred = poly_pred.clamp(min=1e-2)

        return cls_score, poly_pred, centerness

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        mask_preds: List[Tensor],
                        centernesses: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False) -> InstanceList:
        assert len(cls_scores) == len(mask_preds) == len(centernesses)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            mask_pred_list = select_single_mlvl(
                mask_preds, img_id, detach=True)
            centerness_list = select_single_mlvl(
                centernesses, img_id, detach=True)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                mask_pred_list=mask_pred_list,
                centerness_list=centerness_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                mask_pred_list: List[Tensor],
                                centerness_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False) -> InstanceData:
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)
        score_thr = cfg.get('score_thr', 0)

        mlvl_mask_preds = []
        mlvl_centerness = []
        mlvl_scores = []
        mlvl_labels = []

        for level_idx, (cls_score, mask_pred, centerness, priors) in \
                enumerate(zip(cls_score_list, mask_pred_list, centerness_list, mlvl_priors)):
            assert cls_score.size()[-2:] == mask_pred.size()[-2:]

            mask_pred = mask_pred.permute(1, 2, 0).reshape(-1, self.num_rays)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()

            scores, labels, keep_idxs, filtered_results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(mask_pred=mask_pred, priors=priors))

            mask_pred = filtered_results['mask_pred']
            priors = filtered_results['priors']
            centerness = centerness[keep_idxs]

            mask_pred = distance2mask(priors, mask_pred, angles=self.angles, num_rays=self.num_rays, max_shape=img_shape)

            mlvl_mask_preds.append(mask_pred)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_centerness.append(centerness)

        results = InstanceData()
        results.masks = torch.cat(mlvl_mask_preds)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        results.centerness = torch.cat(mlvl_centerness)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            img_meta=img_meta)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            mask_pred = results.masks
            scale_factor = torch.Tensor(scale_factor).to(results.masks.device)
            scale_factor = scale_factor.unsqueeze(0).repeat(self.num_rays, 1)
            scale_factor = scale_factor.unsqueeze(0).repeat(mask_pred.shape[0], 1, 1)
            mask_pred = mask_pred * scale_factor
            results.masks = mask_pred

        centerness = results.pop('centerness')
        results.scores = results.scores * centerness

        if results.masks.numel() > 0:
            bbox_pred = torch.stack([
                mask_pred[..., 0].min(1)[0], mask_pred[..., 1].min(1)[0],
                mask_pred[..., 0].max(1)[0], mask_pred[..., 1].max(1)[0]
            ], dim=-1)

            det_bboxes, keep_idxs = batched_nms(bbox_pred, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
        else:
            results.bboxes = results.scores.new_zeros(len(results.scores), 4)

        # TODO: when testing inference speed, please comment out the lines 257-260 for a fair comparison
        mask_pred = mask2result(
            results.masks, img_meta['ori_shape']
        )
        results.masks = mask_pred

        return results


# test
def distance2mask(points, distances, angles=None, num_rays=36, max_shape=None):
    """Decode distance prediction to 36 mask points
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distances (Tensor): Distance from the given point to 36,from angle 0 to 350.
        angles (Tensor):
        num_rays (int):
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded masks.
    """
    if angles is None:
        angles = torch.arange(0, 360, 360 / num_rays, device=distances.device) / 180 * math.pi

    num_points = points.shape[0]
    points = points[:, :, None].repeat(1, 1, num_rays)
    c_x, c_y = points[:, 0], points[:, 1]

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    sin = sin[None, :].repeat(num_points, 1)
    cos = cos[None, :].repeat(num_points, 1)

    x = distances * cos + c_x
    y = distances * sin + c_y

    if max_shape is not None:
        x = x.clamp(min=0, max=max_shape[1] - 1)
        y = y.clamp(min=0, max=max_shape[0] - 1)

    res = torch.cat([x[:, :, None], y[:, :, None]], dim=2)
    return res


def mask2result(masks, ori_shape):
    """Convert detection results to a list of numpy arrays.

    Args:
        masks (Tensor): shape (n, 2, 36)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    img_h, img_w = ori_shape
    device = masks.device

    mask_results = torch.zeros((masks.shape[0], img_h, img_w))
    for i in range(masks.shape[0]):
        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask = [masks[i].unsqueeze(1).int().data.cpu().numpy()]
        im_mask = cv2.drawContours(im_mask, mask, -1, 1, -1)
        im_mask = torch.from_numpy(im_mask).to(dtype=torch.uint8, device=device)
        mask_results[i] = im_mask
    return mask_results