from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv.image.geometric import _scale_size
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms import RandomResize, RandomChoiceResize

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type


@TRANSFORMS.register_module()
class PolarResize(MMCV_Resize):

    def _resize_masks(self, results: dict) -> None:
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is not None:
            if self.keep_ratio:
                results['gt_masks'] = results['gt_masks'].rescale(
                    results['scale'])
            else:
                results['gt_masks'] = results['gt_masks'].resize(
                    results['img_shape'])

        if results.get('gt_polys', None) is not None:
            if self.keep_ratio:
                results['gt_polys'] = results['gt_polys'].rescale(
                    results['scale'])
            else:
                results['gt_polys'] = results['gt_polys'].resize(
                    results['img_shape'])

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].rescale_(results['scale_factor'])
            if self.clip_object_border:
                results['gt_bboxes'].clip_(results['img_shape'])

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the Resize."""
        w_scale, h_scale = results['scale_factor']
        homography_matrix = np.array(
            [[w_scale, 0, 0], [0, h_scale, 0], [0, 0, 1]], dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._record_homography_matrix(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class PolarRandomFlip(MMCV_RandomFlip):

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the RandomFlip."""
        cur_dir = results['flip_direction']
        h, w = results['img'].shape[:2]

        if cur_dir == 'horizontal':
            homography_matrix = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'vertical':
            homography_matrix = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'diagonal':
            homography_matrix = np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        else:
            homography_matrix = np.eye(3, dtype=np.float32)

        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip polys
        if results.get('gt_polys', None) is not None:
            results['gt_polys'] = results['gt_polys'].flip(
                results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)


@TRANSFORMS.register_module()
class PolarRandomResize(RandomResize):
    def __init__(
            self,
            scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
            ratio_range: Tuple[float, float] = None,
            resize_type: str = 'PolarResize',
            **resize_kwargs,
    ) -> None:
        super().__init__(
            scale=scale,
            ratio_range=ratio_range,
            resize_type=resize_type,
            **resize_kwargs
        )


@TRANSFORMS.register_module()
class PolarRandomChoiceResize(RandomChoiceResize):
    def __init__(
            self,
            scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
            resize_type: str = 'PolarResize',
            **resize_kwargs,
    ) -> None:
        super().__init__(
            scale=scale,
            resize_type=resize_type,
            **resize_kwargs
        )
