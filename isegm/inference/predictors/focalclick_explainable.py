import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms.zoom_in import ZoomIn
from isegm.inference.transforms import (
    AddHorizontalFlip,
    SigmoidForPred,
    LimitLongestSide,
    ResizeTrans,
)
from isegm.utils.crop_local import (
    map_point_in_bbox,
    get_focus_cropv1,
    get_focus_cropv2,
    get_object_crop,
    get_click_crop,
)
import sys
import numpy


class FocalPredictorExplainable(object):
    def __init__(
        self,
        model,
        device,
        net_clicks_limit=None,
        with_flip=False,
        zoom_in=None,
        max_size=None,
        infer_size=512,
        focus_crop_r=1.4,
        ensure_minimum_focus_crop_size=False,
        **kwargs,
    ):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.crop_l = infer_size
        self.focus_crop_r = focus_crop_r
        self.transforms.append(ResizeTrans(self.crop_l, zoom_in.recompute_click_size_on_zoom))
        self.transforms.append(SigmoidForPred())
        self.focus_roi = None
        self.global_roi = None

        if self.with_flip:
            self.transforms.append(AddHorizontalFlip())
            
        self.ensure_minimum_focus_crop_size = ensure_minimum_focus_crop_size

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)

        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def set_prev_mask(self, mask):
        self.prev_prediction = (
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device).float()
        )

        self.transforms[0]._prev_probs = (
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().numpy()
        )

    def get_prediction(self, clicker, prev_mask=None):
        info = {}

        clicks_list = clicker.get_clicks()
        click = clicks_list[-1]
        last_y, last_x = click.coords[0], click.coords[1]
        self.last_y = last_y
        self.last_x = last_x

        if self.click_models is not None:
            model_indx = (
                min(
                    clicker.click_indx_offset + len(clicks_list), len(self.click_models)
                )
                - 1
            )
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)

        info['original_image'] = self.original_image.cpu().numpy()[0]
        info['prev_mask'] = prev_mask.cpu().numpy()[0, 0]
        info['input_image'] = input_image.cpu().numpy()[0]
        info['raw_clicks'] = [clicks_list]

        image_nd, clicks_lists, is_image_changed, roi = self.apply_transforms(
            input_image, [clicks_list]
        )

        info['image_transforms'] = {
            'image_nd': image_nd,
            'clicks_lists': clicks_lists,
            'is_image_changed': is_image_changed,
            'roi': roi,
        }

        try:
            roi = self.transforms[0]._object_roi
            y1, y2, x1, x2 = roi
            global_roi = (y1, y2 + 1, x1, x2 + 1)
        except:
            h, w = prev_mask.shape[-2], prev_mask.shape[-1]
            global_roi = (0, h, 0, w)
        self.global_roi = global_roi

        info['global_roi'] = global_roi

        pred_logits, feature, coarse_seg_info = self._get_prediction(
            image_nd, clicks_lists, is_image_changed
        )
        info['coarse_seg'] = coarse_seg_info
        info['coarse_seg_prediction'] = pred_logits.cpu().numpy()[0, 0]

        prediction = F.interpolate(
            pred_logits, mode='bilinear', align_corners=True, size=image_nd.size()[2:]
        )

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        # self.prev_prediction = prediction
        # return prediction.cpu().numpy()[0, 0]

        prediction = torch.log(prediction / (1 - prediction))
        coarse_mask = prediction

        info['coarse_seg_mask'] = coarse_mask.cpu().numpy()[0, 0]

        prev_mask = prev_mask
        clicks_list = clicker.get_clicks()
        image_full = self.original_image

        coarse_mask_np = coarse_mask.cpu().numpy()[0, 0]
        prev_mask_np = prev_mask.cpu().numpy()[0, 0]
        
        #if minimum crop should be ensured
        min_crop_size = int(self.crop_l / 5) if self.ensure_minimum_focus_crop_size else 0

        # === Ablation Studies for Different Strategies of Focus Crop
        y1, y2, x1, x2 = get_focus_cropv1(
            coarse_mask_np, prev_mask_np, global_roi, last_y, last_x, self.focus_crop_r, min_crop=min_crop_size
        )
        # y1,y2,x1,x2 = get_focus_cropv1(coarse_mask_np,prev_mask_np, global_roi,last_y,last_x, 1.2)
        # y1,y2,x1,x2 = get_focus_cropv1(coarse_mask_np,prev_mask_np, global_roi,last_y,last_x, 1.6)
        # y1,y2,x1,x2 = get_focus_cropv2(coarse_mask_np,prev_mask_np, global_roi,last_y,last_x, 1.4)
        # y1,y2,x1,x2 = get_object_crop(coarse_mask_np,prev_mask_np, global_roi,last_y,last_x, 1.4)
        # y1,y2,x1,x2 = get_click_crop(coarse_mask_np,prev_mask_np, global_roi,last_y,last_x, 1.4)

        focus_roi = (y1, y2, x1, x2)
        self.focus_roi = focus_roi

        info['focus_roi'] = focus_roi

        focus_roi_in_global_roi = self.mapp_roi(focus_roi, global_roi)
        info['focus_roi_in_global_roi'] = focus_roi_in_global_roi

        focus_pred, refine_info = self._get_refine(
            pred_logits,
            image_full,
            clicks_list,
            feature,
            focus_roi,
            focus_roi_in_global_roi,
        )  # .cpu().numpy()[0, 0]

        info['refine_info'] = refine_info

        focus_pred = F.interpolate(
            focus_pred, (y2 - y1, x2 - x1), mode='bilinear', align_corners=True
        )  # .cpu().numpy()[0, 0]

        info['focus_pred_interpolated'] = focus_pred.cpu().numpy()[0, 0]

        if len(clicks_list) > 10:
            # if len(clicks_list) > 1:
            coarse_mask = self.prev_prediction
            coarse_mask = torch.log(coarse_mask / (1 - coarse_mask))

        info['progressive_merge_mask'] = coarse_mask.cpu().numpy()[0, 0]

        coarse_mask[:, :, y1:y2, x1:x2] = focus_pred
        coarse_mask = torch.sigmoid(coarse_mask)

        info['final_pred'] = coarse_mask.cpu().numpy()[0, 0]

        self.prev_prediction = coarse_mask
        self.transforms[0]._prev_probs = coarse_mask.cpu().numpy()
        return coarse_mask.cpu().numpy()[0, 0], info

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        output, info = self.net(image_nd, points_nd)
        return output['instances'], output['feature'], info

    def _get_refine(
        self, coarse_mask, image, clicks, feature, focus_roi, focus_roi_in_global_roi
    ):
        info = {}

        y1, y2, x1, x2 = focus_roi
        image_focus = image[:, :, y1:y2, x1:x2]

        info['image_focus'] = image_focus

        image_focus = F.interpolate(
            image_focus, (self.crop_l, self.crop_l), mode='bilinear', align_corners=True
        )
        info['image_focus_interpolated'] = image_focus

        mask_focus = coarse_mask
        points_nd = self.get_points_nd_inbbox(clicks, y1, y2, x1, x2)
        info['points_nd'] = points_nd

        y1, y2, x1, x2 = focus_roi_in_global_roi
        # y1, y2, x1, x2 = (
        #     focus_roi_in_global_roi
        #     if focus_roi_in_global_roi != (0.0, 256.0, 0.0, 256.0)
        #     else (int(y1 / 2), int(y2 / 2), int(x1 / 2), int(x2 / 2))
        # )
        # when there was no mask found in previous step, but it reappears after coarse segmentation, focus crop for logits/features get's lost as it's supposed to crop the
        # resized feature maps. This hack fixes that, but introduces bugs in general case. Need a good way to detect this situation.

        roi = (
            torch.tensor([0, x1, y1, x2, y2])
            .unsqueeze(0)
            .float()
            .to(image_focus.device)
        )

        info['roi'] = roi

        pred, refine_info = self.net.refine(
            image_focus, points_nd, feature, mask_focus, roi
        )  # ['instances_refined']
        info['refine_pass'] = refine_info

        focus_coarse, focus_refined = (
            pred['instances_coarse'],
            pred['instances_refined'],
        )
        # self.focus_coarse = torch.sigmoid(focus_coarse).cpu().numpy()[0, 0] * 255
        # self.focus_refined = torch.sigmoid(focus_refined).cpu().numpy()[0, 0] * 255
        self.focus_coarse = focus_coarse.cpu().numpy()[0, 0]
        self.focus_refined = focus_refined.cpu().numpy()[0, 0]

        info['focus_coarse'] = self.focus_coarse
        info['focus_refined'] = self.focus_refined

        return focus_refined, info

    def mapp_roi(self, focus_roi, global_roi):
        yg1, yg2, xg1, xg2 = global_roi
        hg, wg = yg2 - yg1, xg2 - xg1
        yf1, yf2, xf1, xf2 = focus_roi

        yf1_n = (yf1 - yg1) * (self.crop_l / hg)
        yf2_n = (yf2 - yg1) * (self.crop_l / hg)
        xf1_n = (xf1 - xg1) * (self.crop_l / wg)
        xf2_n = (xf2 - xg1) * (self.crop_l / wg)

        yf1_n = max(yf1_n, 0)
        yf2_n = min(yf2_n, self.crop_l)
        xf1_n = max(xf1_n, 0)
        xf2_n = min(xf2_n, self.crop_l)
        return (yf1_n, yf2_n, xf1_n, xf2_n)

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)
        print('_set_transform_states')

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        roi = None
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed
            if type(t) is ZoomIn:
                roi = t._object_roi

        return image_nd, clicks_lists, is_image_changed, roi

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [
            sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists
        ]
        num_neg_clicks = [
            len(clicks_list) - num_pos
            for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)
        ]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        # for clicks_list in clicks_lists:
        #     clicks_list = clicks_list[: self.net_clicks_limit]
        #     pos_clicks = [
        #         click.coords_and_indx for click in clicks_list if click.is_positive
        #     ]
        #     pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [
        #         (-1, -1, -1)
        #     ]

        #     neg_clicks = [
        #         click.coords_and_indx for click in clicks_list if not click.is_positive
        #     ]
        #     neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [
        #         (-1, -1, -1)
        #     ]
        #     total_clicks.append(pos_clicks + neg_clicks)
        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.as_tuple for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1, -1)]

            neg_clicks = [click.as_tuple for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_points_nd_inbbox(self, clicks_list, y1, y2, x1, x2):
        total_clicks = []
        num_pos = sum(x.is_positive for x in clicks_list)
        num_neg = len(clicks_list) - num_pos
        num_max_points = max(num_pos, num_neg)
        num_max_points = max(1, num_max_points)
        pos_clicks, neg_clicks = [], []
        for click in clicks_list:
            flag, y, x, index, disk_radius = click.is_positive, click.coords[0], click.coords[1], 0, click.disk_radius
            y, x, disk_radius = map_point_in_bbox(y, x, disk_radius, y1, y2, x1, x2, self.crop_l)
            if flag:
                pos_clicks.append((y, x, index, disk_radius))
            else:
                neg_clicks.append((y, x, index, disk_radius))

        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1, -1)]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)
        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone(),
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
        print('set')
