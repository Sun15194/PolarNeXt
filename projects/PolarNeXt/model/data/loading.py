from typing import Union

import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
# from ..visualizer import visualize_polygon


@TRANSFORMS.register_module()
class PolarLoadAnnotations(MMCV_LoadAnnotations):
    def __init__(
            self,
            with_mask: bool = False,
            poly2mask: bool = True,
            box_type: str = 'hbox',
            **kwargs) -> None:
        super(PolarLoadAnnotations, self).__init__(**kwargs)
        self.with_mask = with_mask
        self.poly2mask = poly2mask
        self.box_type = box_type

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _poly2mask(self, mask_ann: Union[list, dict], img_h: int,
                   img_w: int) -> np.ndarray:
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            np.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _process_masks(self, results: dict) -> list:
        """Process gt_masks and filter invalid polygons.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            list: Processed gt_masks.
        """
        gt_masks = []
        gt_ignore_flags = []
        for i, instance in enumerate(results.get('instances', [])):
            gt_mask = instance['mask']
            # If the annotation of segmentation mask is invalid,
            # ignore the whole instance.
            if isinstance(gt_mask, list):
                gt_mask = [
                    np.array(polygon) for polygon in gt_mask
                    if len(polygon) % 2 == 0 and len(polygon) >= 6
                ]
                if len(gt_mask) == 0:
                    # ignore this instance and set gt_mask to a fake mask
                    instance['ignore_flag'] = 1
                    gt_mask = [np.zeros(6)]
                elif len(gt_mask) > 1:
                    gt_mask = [frag.reshape(-1, 2) for frag in gt_mask]
                    _, point_list = max(
                        (0.5 * np.abs(np.dot(fragment[:, 0], np.roll(fragment[:, 1], 1)) -
                                      np.dot(fragment[:, 1], np.roll(fragment[:, 0], 1))),
                         fragment)
                        for fragment in gt_mask
                    )

                    x_min, y_min = np.min(point_list[:, 0]), np.min(point_list[:, 1])
                    x_max, y_max = np.max(point_list[:, 0]), np.max(point_list[:, 1])

                    # gt_bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                    gt_bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]
                    gt_mask = [point_list.reshape(-1)]

                    results['instances'][i]['bbox'] = gt_bbox
            else:
                # `PolygonMasks` requires a ploygon of format List[np.array],
                # other formats are invalid.
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]

            gt_masks.append(gt_mask)
            # re-process gt_ignore_flags
            gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        return gt_masks

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        h, w = results['ori_shape']
        gt_masks = self._process_masks(results)

        gt_polys = PolygonMasks([mask for mask in gt_masks], h, w)
        gt_masks = BitmapMasks(
            [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)

        results['gt_polys'] = gt_polys
        results['gt_masks'] = gt_masks

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_bbox:
            self._load_bboxes(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str