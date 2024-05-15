from .base import ISTrainer
from isegm.simple_click.inference.predictors import get_predictor

import torch
import random
import numpy as np
import cv2

class SimpleClickTrainer(ISTrainer):
    def __init__(self, layerwise_decay=False, **kwargs):
        super().__init__(**kwargs)
        
        # LAYERWISE DECAY DISABLED FOR NOW
        # if layerwise_decay:
        #     self.optim = get_optimizer_with_layerwise_decay(
        #         model, optimizer, optimizer_params
        #     )
        # else:
        #     self.optim = get_optimizer(model, optimizer, optimizer_params)
    
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

                    points = self._get_next_points(
                        prev_output, orig_gt_mask, points, click_indx + 1
                    )

                    if not validation:
                        self.net.train()

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

            net_input = (
                torch.cat((image, prev_output), dim=1)
                if self.net.with_prev_mask
                else image
            )
            output = self.net(net_input, points)

            loss = 0.0
            loss = self.add_loss(
                'instance_loss',
                loss,
                losses_logging,
                validation,
                lambda: (output['instances'], batch_data['instances']),
            )
            loss = self.add_loss(
                'instance_aux_loss',
                loss,
                losses_logging,
                validation,
                lambda: (output['instances_aux'], batch_data['instances']),
            )

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(
                            *(output.get(x) for x in m.pred_outputs),
                            *(batch_data[x] for x in m.gt_outputs),
                        )
        return loss, losses_logging
    
    def _get_next_points(self, pred, gt, points, click_indx, pred_thresh=0.49):
        assert click_indx > 0
        pred = pred.cpu().numpy()[:, 0, :, :]
        gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

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
                if is_positive:
                    points[bindx, num_points - click_indx, 0] = float(coords[0])
                    points[bindx, num_points - click_indx, 1] = float(coords[1])
                    points[bindx, num_points - click_indx, 2] = float(click_indx)
                else:
                    points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                    points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                    points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)

        return points