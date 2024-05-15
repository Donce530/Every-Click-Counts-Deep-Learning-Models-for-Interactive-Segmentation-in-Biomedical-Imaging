import torch

from typing import List
from isegm.inference.clicker import Click
from isegm.utils.misc import get_bbox_iou, get_bbox_from_mask, expand_bbox, clamp_bbox
from .base import BaseTransform


class ZoomIn(BaseTransform):
    def __init__(
        self,
        target_size=480,
        skip_clicks=1,
        expansion_ratio=1.4,
        min_crop_size=10,  # 200
        recompute_thresh_iou=0.5,
        prob_thresh=0.49,
        optimistic_masking=False,
        recompute_click_size_on_zoom=None, # DYNAMIC CLICK PROP
    ):
        super().__init__()
        self.target_size = target_size
        self.min_crop_size = min_crop_size
        self.skip_clicks = skip_clicks
        self.expansion_ratio = expansion_ratio
        self.recompute_thresh_iou = recompute_thresh_iou
        self.prob_thresh = prob_thresh
        self.optimistic_masking = optimistic_masking
        self.recompute_click_size_on_zoom = recompute_click_size_on_zoom
        
        if self.recompute_click_size_on_zoom is None:
            raise Exception(f"recompute_click_size_on_zoom is undefined. It should be a boolean value, predictor needs to know if click is dynamic or not")

        self._input_image_shape = None
        self._prev_probs = None
        self._object_roi = None
        self._roi_image = None

    def transform(self, image_nd, clicks_lists: List[List[Click]]):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        self.image_changed = False

        clicks_list = clicks_lists[0]
        if len(clicks_list) <= self.skip_clicks:
            return image_nd, clicks_lists

        self._input_image_shape = image_nd.shape

        current_object_roi = None
        if self._prev_probs is not None:
            current_pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
            if current_pred_mask.sum() > 0:
                current_object_roi = get_object_roi(
                    current_pred_mask,
                    clicks_list,
                    self.expansion_ratio,
                    self.min_crop_size,
                )
            # NEW OPTIMISTIC MASKING
            # else:
            #     if self.optimistic_masking and self._prev_probs.sum() > 0:
            #         optimistic_treshold = self._prev_probs.max() * 0.75
            #         current_pred_mask = (self._prev_probs > optimistic_treshold)[0, 0]
            #     current_object_roi = get_object_roi(
            #         current_pred_mask,
            #         clicks_list,
            #         self.expansion_ratio,
            #         self.min_crop_size,
            #     )

        else:
            print('None')

        if current_object_roi is None:
            if self.skip_clicks >= 0:
                return image_nd, clicks_lists
            else:
                current_object_roi = 0, image_nd.shape[2] - 1, 0, image_nd.shape[3] - 1

        # here
        update_object_roi = True
        if self._object_roi is None:
            update_object_roi = True
        elif not check_object_roi(self._object_roi, clicks_list):
            update_object_roi = True
        elif (
            get_bbox_iou(current_object_roi, self._object_roi)
            < self.recompute_thresh_iou
        ):
            update_object_roi = True

        if update_object_roi:
            self._object_roi = current_object_roi
            self.image_changed = True
        self._roi_image = get_roi_image_nd(image_nd, self._object_roi, self.target_size)

        tclicks_lists = [self._transform_clicks(clicks_list)]
        return self._roi_image.to(image_nd.device), tclicks_lists

    # def transform(self, image_nd, clicks_lists: List[List[Click]]):
    #     print("Entering transform...")

    #     assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
    #     self.image_changed = False

    #     clicks_list = clicks_lists[0]
    #     if len(clicks_list) <= self.skip_clicks:
    #         print("Exiting due to insufficient clicks...")
    #         return image_nd, clicks_lists

    #     self._input_image_shape = image_nd.shape

    #     current_object_roi = None
    #     if self._prev_probs is not None:
    #         print("Previous probabilities exist...")
    #         current_pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
    #         if current_pred_mask.sum() > 0:
    #             print("Calculating current_object_roi based on previous mask...")
    #             current_object_roi = get_object_roi(
    #                 current_pred_mask,
    #                 clicks_list,
    #                 self.expansion_ratio,
    #                 self.min_crop_size,
    #             )
    #         else:
    #             print("Previous mask has no positive values after thresholding...")
    #     else:
    #         print("No previous probabilities found...")

    #     if current_object_roi is None:
    #         if self.skip_clicks >= 0:
    #             print(
    #                 "Exiting since current_object_roi is None and skip_clicks is >= 0..."
    #             )
    #             return image_nd, clicks_lists
    #         else:
    #             print("Setting current_object_roi to the full image dimensions...")
    #             current_object_roi = 0, image_nd.shape[2] - 1, 0, image_nd.shape[3] - 1

    #     # here
    #     update_object_roi = True
    #     if self._object_roi is None:
    #         print("No previous _object_roi exists...")
    #     elif not check_object_roi(self._object_roi, clicks_list):
    #         print("_object_roi doesn't cover the latest positive click...")
    #     elif (
    #         get_bbox_iou(current_object_roi, self._object_roi)
    #         < self.recompute_thresh_iou
    #     ):
    #         print("The IoU between current and previous ROIs is below threshold...")

    #     if update_object_roi:
    #         print("Updating _object_roi and setting image_changed to True...")
    #         self._object_roi = current_object_roi
    #         self.image_changed = True

    #     print("Fetching the ROI image based on _object_roi...")
    #     self._roi_image = get_roi_image_nd(image_nd, self._object_roi, self.target_size)

    #     tclicks_lists = [self._transform_clicks(clicks_list)]
    #     return self._roi_image.to(image_nd.device), tclicks_lists

    def inv_transform(self, prob_map):
        if self._object_roi is None:
            self._prev_probs = prob_map.cpu().numpy()
            return prob_map

        assert prob_map.shape[0] == 1
        rmin, rmax, cmin, cmax = self._object_roi
        prob_map = torch.nn.functional.interpolate(
            prob_map,
            size=(rmax - rmin + 1, cmax - cmin + 1),
            mode='bilinear',
            align_corners=True,
        )

        if self._prev_probs is not None:
            new_prob_map = torch.zeros(
                *self._prev_probs.shape, device=prob_map.device, dtype=prob_map.dtype
            )
            new_prob_map[:, :, rmin : rmax + 1, cmin : cmax + 1] = prob_map
            # new_prob_map[:, :, rmin:rmax, cmin:cmax] = prob_map
        else:
            new_prob_map = prob_map

        self._prev_probs = new_prob_map.cpu().numpy()

        return new_prob_map

    def check_possible_recalculation(self):
        if (
            self._prev_probs is None
            or self._object_roi is not None
            or self.skip_clicks > 0
        ):
            return False

        pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
        if pred_mask.sum() > 0:
            possible_object_roi = get_object_roi(
                pred_mask, [], self.expansion_ratio, self.min_crop_size
            )
            image_roi = (
                0,
                self._input_image_shape[2] - 1,
                0,
                self._input_image_shape[3] - 1,
            )
            if get_bbox_iou(possible_object_roi, image_roi) < 0.50:
                return True
        return False

    def get_state(self):
        roi_image = self._roi_image.cpu() if self._roi_image is not None else None
        return (
            self._input_image_shape,
            self._object_roi,
            self._prev_probs,
            roi_image,
            self.image_changed,
        )

    def set_state(self, state):
        (
            self._input_image_shape,
            self._object_roi,
            self._prev_probs,
            self._roi_image,
            self.image_changed,
        ) = state

    def reset(self):
        self._input_image_shape = None
        self._object_roi = None
        self._prev_probs = None
        self._roi_image = None
        self.image_changed = False

    # def _transform_clicks(self, clicks_list):
    #     if self._object_roi is None:
    #         return clicks_list

    #     rmin, rmax, cmin, cmax = self._object_roi
    #     crop_height, crop_width = self._roi_image.shape[2:]

    #     transformed_clicks = []
    #     for click in clicks_list:
    #         new_r = crop_height * (click.coords[0] - rmin) / (rmax - rmin + 1)
    #         new_c = crop_width * (click.coords[1] - cmin) / (cmax - cmin + 1)
    #         transformed_clicks.append(click.copy(coords=(new_r, new_c)))
    #     return transformed_clicks
    
    def _transform_clicks(self, clicks_list):
        if self._object_roi is None:
            return clicks_list

        rmin, rmax, cmin, cmax = self._object_roi
        
        roi_height = rmax - rmin
        roi_width = cmax - cmin

        if roi_height == 0:
            roi_height = 1
        if roi_width == 0:
            roi_width = 1
        
        crop_height, crop_width = self._roi_image.shape[2:]
        
        scale_r = crop_height / roi_height
        scale_c = crop_width / roi_width
        avg_scale = (scale_r + scale_c) / 2

        transformed_clicks = []
        # print(f'Transforming {len(clicks_list)} clicks...')
        # print(f'ROI height: {roi_height}, ROI width: {roi_width}')
        # print(f'Crop height: {crop_height}, Crop width: {crop_width}')
        # print(f'Scale r: {scale_r}, Scale c: {scale_c}, Avg scale: {avg_scale}')
        # print(f'Recompute click size on zoom: {self.recompute_click_size_on_zoom}')
        for click in clicks_list:
            # print(f'Old click: {click.as_tuple}')
            new_r = (click.coords[0] - rmin) * scale_r
            new_c = (click.coords[1] - cmin) * scale_c
            
            if self.recompute_click_size_on_zoom:
                transformed_radius = click.disk_radius * avg_scale
            else:
                transformed_radius = click.disk_radius
            
            transformed_click = click.copy(coords=(new_r, new_c), disk_radius=transformed_radius)
            # print(f'New click: {transformed_click.as_tuple}')
            transformed_clicks.append(transformed_click)
        return transformed_clicks


def get_object_roi(pred_mask, clicks_list, expansion_ratio, min_crop_size):
    pred_mask = pred_mask.copy()

    for click in clicks_list:
        if click.is_positive:
            pred_mask[int(click.coords[0]), int(click.coords[1])] = 1

    bbox = get_bbox_from_mask(pred_mask)
    bbox = expand_bbox(bbox, expansion_ratio, min_crop_size)
    h, w = pred_mask.shape[0], pred_mask.shape[1]
    bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

    return bbox


def get_roi_image_nd(image_nd, object_roi, target_size):
    rmin, rmax, cmin, cmax = object_roi

    height = rmax - rmin + 1
    width = cmax - cmin + 1

    if isinstance(target_size, tuple):
        new_height, new_width = target_size
    else:
        scale = target_size / max(height, width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))

    with torch.no_grad():
        roi_image_nd = image_nd[:, :, rmin : rmax + 1, cmin : cmax + 1]

    return roi_image_nd


def check_object_roi(object_roi, clicks_list):
    for click in clicks_list:
        if click.is_positive:
            if click.coords[0] < object_roi[0] or click.coords[0] >= object_roi[1]:
                return False
            if click.coords[1] < object_roi[2] or click.coords[1] >= object_roi[3]:
                return False

    return True
