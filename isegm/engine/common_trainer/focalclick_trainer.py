from .base import ISTrainer

import torch
import random
import numpy as np
import cv2

class FocalClickTrainer(ISTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, gt_mask, points = (
                batch_data['images'],
                batch_data['instances'],
                batch_data['points'],
            )
            points_focus = batch_data['points_focus']
            rois = batch_data['rois']
            orig_image, orig_gt_mask, orig_points = (
                image.clone(),
                gt_mask.clone(),
                points.clone(),
            )

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            last_click_indx = None

            min_num_iters = 0

            with torch.no_grad():
                num_iters = random.randint(min_num_iters, self.max_num_next_clicks)

                for click_indx in range(num_iters):
                    last_click_indx = click_indx

                    if not validation:
                        self.net.eval()

                    if self.click_models is None or click_indx >= len(
                        self.click_models
                    ):
                        eval_model = self.net
                    else:
                        eval_model = self.click_models[click_indx]

                    net_input = (
                        torch.cat((image, prev_output), dim=1)
                        if self.net.with_prev_mask
                        else image
                    )

                    prev_output = torch.sigmoid(
                        eval_model(net_input, points)['instances']
                    )

                    points, points_focus = self._get_next_points_removeall(
                        prev_output,
                        orig_gt_mask,
                        points,
                        points_focus,
                        rois,
                        click_indx + 1,
                    )

                    if not validation:
                        self.net.train()
                        # for m in self.net.feature_extractor.modules():
                        #        m.eval()

                if (
                    self.net.with_prev_mask
                    and self.prev_mask_drop_prob > 0
                    and last_click_indx is not None
                ):
                    zero_mask = (
                        np.random.random(size=prev_output.size(0))
                        < self.prev_mask_drop_prob
                    )
                    prev_output[zero_mask] = torch.zeros_like(prev_output[zero_mask])

            batch_data['points'] = points
            batch_data['prev_mask'] = prev_output
            batch_data['points_focus'] = points_focus

            net_input = (
                torch.cat((image, prev_output), dim=1)
                if self.net.with_prev_mask
                else image
            )
            output = self.net(net_input, points)

            # ====== refine =====
            images_focus, points_focus, rois = (
                batch_data['images_focus'],
                batch_data['points_focus'],
                batch_data['rois'],
            )
            full_feature, full_logits = output['feature'], output['instances']
            bboxes = torch.chunk(rois, rois.shape[0], dim=0)
            # print( len(bboxes), bboxes[0].shape, rois.shape  )
            refine_output = self.net.refine(
                images_focus, points_focus, full_feature, full_logits, bboxes
            )

            loss = 0.0
            loss = self.add_loss(
                'instance_loss',
                loss,
                losses_logging,
                validation,
                lambda: (output['instances'], batch_data['instances']),
            )

            loss = self.add_loss(
                'instance_click_loss',
                loss,
                losses_logging,
                validation,
                lambda: (
                    output['instances'],
                    batch_data['instances'],
                    output['click_map'],
                ),
            )

            loss = self.add_loss(
                'instance_aux_loss',
                loss,
                losses_logging,
                validation,
                lambda: (output['instances_aux'], batch_data['instances']),
            )

            loss = self.add_loss(
                'trimap_loss',
                loss,
                losses_logging,
                validation,
                lambda: (refine_output['trimap'], batch_data['trimap_focus']),
            )

            loss = self.add_loss(
                'instance_refine_loss',
                loss,
                losses_logging,
                validation,
                lambda: (
                    refine_output['instances_refined'],
                    batch_data['instances_focus'],
                    batch_data['trimap_focus'],
                ),
            )

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(
                            *(output.get(x) for x in m.pred_outputs),
                            *(batch_data[x] for x in m.gt_outputs),
                        )
        return (
            loss,
            losses_logging
        )
        
    def _get_next_points_removeall(self, pred, gt, points, points_focus, rois, click_indx, pred_thresh=0.49, remove_prob=0.0):
        assert click_indx > 0
        pred = pred.cpu().numpy()[:, 0, :, :]
        gt = gt.cpu().numpy()[:, 0, :, :] > 0.5
        rois = rois.cpu().numpy()
        h, w = gt.shape[-2], gt.shape[-1]

        fn_mask = np.logical_and(gt, pred < pred_thresh)
        fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

        fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
        fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
        num_points = points.size(1) // 2
        points = points.clone()

        for bindx in range(fn_mask.shape[0]):
            fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
            fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

            fn_max_dist = np.max(fn_mask_dt)
            fp_max_dist = np.max(fp_mask_dt)

            is_positive = fn_max_dist > fp_max_dist
            dt = fn_mask_dt if is_positive else fp_mask_dt
            inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
            indices = np.argwhere(inner_mask)
            if len(indices) > 0:
                coords = indices[np.random.randint(0, len(indices))]
                if np.random.rand() < remove_prob:
                    points[bindx] = points[bindx] * 0.0 - 1.0
                if is_positive:
                    points[bindx, num_points - click_indx, 0] = float(coords[0])
                    points[bindx, num_points - click_indx, 1] = float(coords[1])
                    points[bindx, num_points - click_indx, 2] = float(click_indx)
                    points[bindx, num_points - click_indx, 3] = float(5) # static disk radius
                else:
                    points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                    points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                    points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)
                    points[bindx, num_points - click_indx, 3] = float(5) # static disk radius

            x1, y1, x2, y2 = rois[bindx]
            point_focus = points[bindx]
            hc, wc = y2 - y1, x2 - x1
            ry, rx = h / hc, w / wc
            bias = torch.tensor([y1, x1, 0, 0]).to(points.device)
            ratio = torch.tensor([ry, rx, 1, 1]).to(points.device)
            points_focus[bindx] = (point_focus - bias) * ratio
        return points, points_focus