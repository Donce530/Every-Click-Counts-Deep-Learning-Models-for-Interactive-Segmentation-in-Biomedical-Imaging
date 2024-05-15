import os
import random
import logging
from copy import deepcopy
from collections import defaultdict

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf

from isegm.ritm.utils.log import logger, TqdmToLogger
from isegm.utils.misc import save_checkpoint
from isegm.ritm.utils.serialization import get_config_repr
from isegm.ritm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from isegm.ritm.inference.iterative_evaluation_training import (
    evaluate_dataset as iterative_evaluate_dataset,
)
from isegm.ritm.inference.predictors import get_predictor
from .optimizer import get_optimizer

import wandb


class ISTrainer(object):
    def __init__(
        self,
        model,
        cfg,
        model_cfg,
        loss_cfg,
        trainset,
        valset,
        device,
        optimizer='adam',
        optimizer_params=None,
        image_dump_interval=200,
        checkpoint_interval=10,
        tb_dump_period=25,
        max_interactive_points=0,
        lr_scheduler=None,
        metrics=None,
        additional_val_metrics=None,
        net_inputs=('images', 'points'),
        max_num_next_clicks=0,
        click_models=None,
        prev_mask_drop_prob=0.0,
        metric_for_best_model='IoU',
        iterative_evaluation_interval=0,
    ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks

        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.metric_for_best_model = metric_for_best_model
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)
        self.best_val_metric_value = 0.0
        self.best_iterative_error = np.inf

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''

        self.trainset = trainset
        self.valset = valset

        logger.info(
            f"Dataset of {trainset.get_samples_number()} samples was loaded for training."
        )
        logger.info(
            f"Dataset of {valset.get_samples_number()} samples was loaded for validation."
        )

        self.train_data = DataLoader(
            trainset,
            cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=False,
            pin_memory=True,
            num_workers=cfg.workers,
        )

        self.val_data = DataLoader(
            valset,
            cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=False,
            pin_memory=True,
            num_workers=cfg.workers,
        )

        self.val_dataset = valset

        self.optim = get_optimizer(model, optimizer, optimizer_params)
        model = self._load_weights(model)

        if cfg.multi_gpu:
            model = get_dp_wrapper(cfg.distributed)(
                model, device_ids=cfg.gpu_ids, output_device=cfg.gpu_ids[0]
            )

        if self.is_master:
            # logger.info(model)
            logger.info(get_config_repr(model._config))

        self.device = device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

        self.iterative_evaluation_interval = iterative_evaluation_interval

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        with wandb.init(
            project=self.cfg.wandb.project_name,
            name=f"{self.model_cfg.name}-{self.cfg.dataset.train}-{self.cfg.exp_name}",
            config=OmegaConf.to_container(self.cfg, resolve=True),
        ):
            logger.info(f'Starting Epoch: {start_epoch}')
            logger.info(f'Total Epochs: {num_epochs}')
            for epoch in range(start_epoch, num_epochs):
                self.training(epoch)
                if validation:
                    self.validation(epoch)
                    if (
                        self.iterative_evaluation_interval > 0
                        and epoch % self.iterative_evaluation_interval == 0
                    ):
                        self.iterative_evaluation(epoch)

    def training(self, epoch):
        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        tbar = (
            tqdm(self.train_data, file=self.tqdm_out, ncols=100)
            if self.is_master
            else self.train_data
        )

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()

        train_loss = 0

        for i, batch_data in enumerate(tbar):
            global_step = epoch * (len(self.train_data) + len(self.val_data)) + i

            loss, losses_logging, splitted_batch_data, outputs = self.batch_forward(
                batch_data
            )

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses_logging["overall"] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging["overall"].item()

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    wandb.log(
                        {
                            f"losses/{loss_name}": loss_value.item(),
                            "global_step": global_step,
                            "epoch": epoch,
                        }
                    )

                tbar.set_description(
                    f"Epoch {epoch}, training loss {train_loss/(i+1):.4f}"
                )

                lr = (
                    self.lr
                    if not hasattr(self, 'lr_scheduler')
                    else self.lr_scheduler.get_lr()[-1]
                )
                wandb.log(
                    {
                        "states/learning_rate": lr,
                        "global_step": global_step,
                        "epoch": epoch,
                    }
                )

        if self.is_master:
            for metric in self.train_metrics:
                metric.log_states(epoch)

            save_checkpoint(
                self.net,
                self.cfg.data_paths.CHECKPOINTS_PATH,
                best=False,
                multi_gpu=self.cfg.multi_gpu,
            )

        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()

    def validation(self, epoch):
        tbar = (
            tqdm(self.val_data, file=self.tqdm_out, ncols=100)
            if self.is_master
            else self.val_data
        )

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()

        for i, batch_data in enumerate(tbar):
            global_step = (
                epoch * len(self.val_data) + (epoch + 1) * len(self.train_data) + i
            )
            (
                loss,
                batch_losses_logging,
                splitted_batch_data,
                outputs,
            ) = self.batch_forward(batch_data, validation=True)

            batch_losses_logging["overall"] = loss
            reduce_loss_dict(batch_losses_logging)

            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())
                wandb.log(
                    {
                        f"val_losses/{loss_name}": loss_value.item(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }
                )

            val_loss += batch_losses_logging["overall"].item()

            if self.is_master:
                tbar.set_description(
                    f"Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}"
                )

        if self.is_master:
            for metric in self.val_metrics:
                metric.log_states(epoch, tag_prefix='val')

            for metric in self.val_metrics:
                if metric.name == self.metric_for_best_model:
                    metric_value = metric.get_epoch_value()
                    if metric_value > self.best_val_metric_value:
                        logger.info(
                            f'New best model with {metric.name}: {metric_value} at epoch {epoch}'
                        )
                        self.best_val_metric_value = metric_value
                        save_checkpoint(
                            self.net,
                            self.cfg.data_paths.CHECKPOINTS_PATH,
                            best=True,
                            multi_gpu=self.cfg.multi_gpu,
                        )
                    break

    def iterative_evaluation(self, epoch):
        validation_set = self.val_dataset
        with torch.no_grad():
            predictor = get_predictor(
                self.net,
                'NoBRS',
                self.device,
            )
            avg_ious, noc, nof, iterative_error = iterative_evaluate_dataset(
                validation_set, predictor, logger
            )
            sample_count = len(validation_set)
            wandb.log(
                {
                    'iterative_evaluation_val/Iterative IoU Error': iterative_error,
                    "iterative_evaluation_val/NoC_85": noc[0],
                    "iterative_evaluation_val/NoC_90": noc[1],
                    "iterative_evaluation_val/NoC_95": noc[2],
                    "iterative_evaluation_val/NoF_85": nof[0],
                    "iterative_evaluation_val/NoF_90": nof[1],
                    "iterative_evaluation_val/NoF_95": nof[2],
                    "iterative_evaluation_val/NoF%_85": nof[0] / sample_count,
                    "iterative_evaluation_val/NoF%_90": nof[1] / sample_count,
                    "iterative_evaluation_val/NoF%_95": nof[2] / sample_count,
                    "iterative_evaluation_val/1st_click_IoU": avg_ious[0],
                    "iterative_evaluation_val/2nd_click_IoU": avg_ious[1],
                    "iterative_evaluation_val/3rd_click_IoU": avg_ious[2],
                    "iterative_evaluation_val/4th_click_IoU": avg_ious[3],
                    "iterative_evaluation_val/5th_click_IoU": avg_ious[4],
                    "iterative_evaluation_val/10th_click_IoU": avg_ious[9],
                    "iterative_evaluation_val/15th_click_IoU": avg_ious[14],
                    "iterative_evaluation_val/20th_click_IoU": avg_ious[19],
                    "epoch": epoch,
                }
            )
            
        if iterative_error < self.best_iterative_error:
            self.best_iterative_error = iterative_error
            logger.info(
                f'New best iterative eval model with error: {iterative_error} at epoch {epoch}'
                )
            save_checkpoint(
                self.net,
                self.cfg.data_paths.CHECKPOINTS_PATH,
                prefix='iterative',
                best=True,
                multi_gpu=self.cfg.multi_gpu,
            )

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, gt_mask, points = (
                batch_data["images"],
                batch_data["instances"],
                batch_data["points"],
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
                        eval_model(net_input, points)["instances"]
                    )

                    points = get_next_points(
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

            batch_data["points"] = points

            net_input = (
                torch.cat((image, prev_output), dim=1)
                if self.net.with_prev_mask
                else image
            )
            output = self.net(net_input, points)

            loss = 0.0
            loss = self.add_loss(
                "instance_loss",
                loss,
                losses_logging,
                validation,
                lambda: (output["instances"], batch_data["instances"]),
            )
            loss = self.add_loss(
                "instance_aux_loss",
                loss,
                losses_logging,
                validation,
                lambda: (output["instances_aux"], batch_data["instances"]),
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
            losses_logging,
            batch_data,
            output,
        )

    def add_loss(
        self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs
    ):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + "_weight", 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(
                self.cfg.data_paths.CHECKPOINTS_PATH.glob(
                    f"{self.cfg.resume_prefix}*.pth"
                )
            )
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f"Load checkpoint from path: {checkpoint_path}")
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.local_rank == 0


def get_next_points(pred, gt, points, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), "constant").astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), "constant").astype(np.uint8)
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


def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location="cpu")["state_dict"]
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)
