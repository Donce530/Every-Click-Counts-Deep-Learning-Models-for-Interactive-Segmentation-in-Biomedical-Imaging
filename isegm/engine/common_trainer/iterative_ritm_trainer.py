from .base import ISTrainer
from isegm.inference.clicker import Clicker, DynamicClicker
import torch.nn.functional as F
from isegm.data.points_sampler import generate_probs

import torch
import numpy as np
import time

import wandb

class IterativeRITMTrainer(ISTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._point_probs = generate_probs(self.max_interactive_points, 0.7)

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()
        
        with torch.set_grad_enabled(not validation):
            time_start = time.time()
            image, gt_mask = (
                batch_data["images"],
                batch_data["instances"],
            )
            image = image.to(self.device)
            gt_mask = gt_mask.to(self.device)
            time_data_moved_to_gpu = time.time()
            
            if self.cfg.one_click_only:
                total_num_clicks = 1 # ONE CLICK
            else:
                total_num_clicks = np.random.choice(np.arange(1, self.max_interactive_points + 1), p=self._point_probs)
            wandb.log(
                    {
                        f"num_clicks": total_num_clicks
                    }
                )

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            
            clickers = []
            for gt in gt_mask:
                clicker = DynamicClicker(gt_mask=gt.squeeze(0).cpu(),
                                         mode=self.clicker_config['mode'],
                                         size_range_modifier=self.clicker_config['size_range_modifier'])
                clickers.append(clicker)
                
            time_clickers_set_up = time.time()
            
            last_click_indx = None
            num_iters = total_num_clicks - 1 # leaves one more click for backprop

            iteration_timing = []
            with torch.no_grad():
                iteration_start_time = time.time()
                for click_indx in range(num_iters):
                    
                    iter_start = time.time()
                    
                    last_click_indx = click_indx
                    
                    if not validation:
                        self.net.eval()

                    if self.click_models is None or click_indx >= len(
                        self.click_models
                    ):
                        eval_model = self.net
                    else:
                        eval_model = self.click_models[click_indx]
                        
                    iter_point_construct_start = time.time()

                    iter_points = torch.empty((0, 2 * (click_indx + 1), 4))
                    for clicker, prev_prediction in zip(clickers, prev_output):
                        clicker.make_next_click(prev_prediction.squeeze(0).cpu().numpy() > 0.5)
                        current_points = self._get_points_from_clicks(clicker)
                        iter_points = torch.cat((iter_points, current_points), dim=0)
                    iter_points = iter_points.to(self.device)
                    
                    iter_point_construct_end = time.time()

                    net_input = (
                        torch.cat((image, prev_output), dim=1)
                        if self.net.with_prev_mask
                        else image
                    )

                    out = eval_model(net_input, iter_points)
                    prev_output = torch.sigmoid(
                         out["instances"]
                    )

                    if not validation:
                        self.net.train()
                    
                    iter_inference_end = time.time()
                
                    iteration_timing.append({
                        'iter_duration': iter_inference_end - iter_start,
                        'iter_pick_model': iter_point_construct_start - iter_start,
                        'iter_point_construct': iter_point_construct_end - iter_point_construct_start,
                        'iter_inference': iter_inference_end - iter_point_construct_end
                    })
                    
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
                    
            iteration_end_time = time.time()

            backprop_points = torch.empty((0, 2 * total_num_clicks, 4))
            for clicker, prev_prediction in zip(clickers, prev_output):
                clicker.make_next_click(prev_prediction.squeeze(0).cpu().numpy() > 0.5)
                current_points = self._get_points_from_clicks(clicker)
                backprop_points = torch.cat((backprop_points, current_points), dim=0)
            backprop_points = backprop_points.to(self.device)
            
            time_backprop_points_ready = time.time()

            # gt_mask_inverse = torch.logical_not(gt_mask)
            
            # net_input = (
            #     torch.cat((image, gt_mask, gt_mask_inverse, prev_output), dim=1)
            #     if self.net.with_prev_mask
            #     else image
            # )
            
            net_input = (
                torch.cat((image, prev_output), dim=1)
                if self.net.with_prev_mask
                else image
            )
            
            # backprop_points = torch.full(batch_data['points'].shape, -1).to(self.device)

            output = self.net(net_input, backprop_points)

            loss = 0.0
            loss = self.add_loss(
                "instance_loss",
                loss,
                losses_logging,
                validation,
                lambda: (output["instances"], gt_mask),
            )
            loss = self.add_loss(
                "instance_aux_loss",
                loss,
                losses_logging,
                validation,
                lambda: (output["instances_aux"], gt_mask),
            )
            
            
            # downsized_masks = F.interpolate(gt_mask, size=(int(gt_mask.shape[-2] / 4), int(gt_mask.shape[-1] / 4)), mode='nearest') #4x smaller than input
            # loss = self.add_loss(
            #     "pre_stage_features_loss",
            #     loss,
            #     losses_logging,
            #     validation,
            #     lambda: (output["pre_stage_features"], downsized_masks),
            # )
            
            time_backprop_done = time.time()
            
            self.logger.info(f'batch_forward() duration: {time_backprop_done - time_start}, moved_to_gpu: {time_data_moved_to_gpu - time_start}, clicker_prep: {time_clickers_set_up - time_data_moved_to_gpu}, {num_iters} inference iterations: {iteration_end_time - iteration_start_time}, backprop points prep: {time_backprop_points_ready - iteration_end_time}, backprop inference pass: {time_backprop_done - time_backprop_points_ready}')
            self.logger.info(f'Per batch timing')
            for i, timing in enumerate(iteration_timing):
                self.logger.info(f'{i}: {iteration_timing[i]}')
            
            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(
                            *(output.get(x).cpu() for x in m.pred_outputs),
                            *(batch_data[x] for x in m.gt_outputs),
                        )
        return (
            loss,
            losses_logging
        )
    
    def _get_points_from_clicks(self, clicker):
        clicks_list = clicker.get_clicks()
        total_clicks = []
        num_pos_clicks = sum(x.is_positive for x in clicks_list)
        num_neg_clicks = len(clicks_list) - num_pos_clicks
        num_max_points = num_pos_clicks + num_neg_clicks
        num_max_points = max(1, num_max_points)

        pos_clicks = [click.as_tuple for click in clicks_list if click.is_positive]
        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1, -1)]
        neg_clicks = [click.as_tuple for click in clicks_list if not click.is_positive]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)
        return torch.tensor(total_clicks)