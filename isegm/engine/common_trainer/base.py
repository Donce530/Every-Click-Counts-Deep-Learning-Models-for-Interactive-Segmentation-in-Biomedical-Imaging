import os
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

from isegm.utils.log import logger, TqdmToLogger
from isegm.utils.misc import save_checkpoint
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from isegm.inference.iterative_evaluation_training import (
    evaluate_dataset as iterative_evaluate_dataset,
)
from isegm.inference.utils import get_predictor, get_dataset
from .optimizer import get_optimizer
from torch.cuda.amp import autocast as autocast, GradScaler
import time

scaler = GradScaler()


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
        clicker_config,
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
        early_stopping_patience=1000,
    ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks
        self.logger = logger

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
        self.best_val_loss = np.inf
        self.best_iterative_error = np.inf
        
        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''

        self.trainset = trainset
        self.valset = valset

        logger.info(
            f'Dataset of {trainset.get_samples_number()} samples was loaded for training.'
        )
        logger.info(
            f'Dataset of {valset.get_samples_number()} samples was loaded for validation.'
        )

        self.train_data = DataLoader(
            trainset,
            cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=False,
            pin_memory=True,
            num_workers=cfg.workers,
        )
        
        logger.info(f'Train data batch dimensions:')
        first_batch = next(iter(self.train_data))
        for key, value in first_batch.items():
            logger.info(f"{key} dimensions: {value.size()}")

        self.val_data = DataLoader(
            valset,
            cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=False,
            pin_memory=True,
            num_workers=cfg.workers,
        )
        
        logger.info(f'Val data batch dimensions:')
        first_batch = next(iter(self.val_data))
        for key, value in first_batch.items():
            logger.info(f"{key} dimensions: {value.size()}")

        self.val_dataset = valset

        self.optim = get_optimizer(model, optimizer, optimizer_params)
        # model = self._load_weights(model)

        if cfg.multi_gpu:
            model = get_dp_wrapper(cfg.distributed)(
                model, device_ids=cfg.gpu_ids, output_device=cfg.gpu_ids[0]
            )

        # if self.is_master:
            # logger.info(model)
            # logger.info(get_config_repr(model._config))

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
        self.clicker_config = clicker_config
        
        self.early_stopping_counter = 0
        self.early_stopping_patience = early_stopping_patience

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
                    if hasattr(self, 'lr_scheduler') and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(self.best_val_metric_value)  # Step LR scheduler
                    if (
                        self.iterative_evaluation_interval > 0
                        and epoch % self.iterative_evaluation_interval == 0
                    ):
                        self.iterative_evaluation(epoch)
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break

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
        
        
        batch_forward_time = 0
        optimizer_time = 0
        wandb_logging_time = 0
        total_duration_start = time.time()

        for i, batch_data in enumerate(tbar):
            global_step = epoch * (len(self.train_data) + len(self.val_data)) + i
            
            checkpoint_a = time.time()
            loss, losses_logging = self.batch_forward(
                batch_data,
                validation=False,
            )
            checkpoint_b = time.time()
            batch_forward_time += checkpoint_b - checkpoint_a
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            checkpoint_c = time.time()
            optimizer_time += checkpoint_c - checkpoint_b

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            for loss_name, loss_value in losses_logging.items():
                wandb.log(
                    {
                        f"losses/{loss_name}": loss_value.item(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }
                )

            train_loss += losses_logging['overall'].item()

            if self.is_master:
                tbar.set_description(
                    f'Epoch {epoch}, training loss {train_loss/(i+1):.4f}'
                )
            checkpoint_d = time.time()
            wandb_logging_time += checkpoint_d - checkpoint_c
                
        total_duration_end = time.time()
        total_time = total_duration_end - total_duration_start
        data_loading_time = total_time - batch_forward_time - optimizer_time - wandb_logging_time
        logger.info(f'Batch duration: {total_time} - forward: {batch_forward_time}, optimizer: {optimizer_time}, wandb: {wandb_logging_time}, data: {data_loading_time}')

        if self.is_master:
            
            if hasattr(self, 'lr_scheduler'):
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # For ReduceLROnPlateau, get learning rate from the optimizer
                    lr = self.optim.param_groups[0]['lr']
                else:
                    # For other schedulers, use the get_lr() method
                    lr = self.lr_scheduler.get_lr()[-1]
            else:
                lr = self.lr
            
            wandb.log(
                {"states/learning_rate": lr, "global_step": global_step, "epoch": epoch}
            )
            
            for metric in self.train_metrics:
                metric.log_states(epoch)

            save_checkpoint(
                self.net,
                self.cfg.data_paths.CHECKPOINTS_PATH,
                best=False,
                multi_gpu=self.cfg.multi_gpu,
            )
            
        if hasattr(self, 'lr_scheduler') and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.MultiStepLR):
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
            ) = self.batch_forward(batch_data, validation=True)

            batch_losses_logging['overall'] = loss
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

            val_loss += batch_losses_logging['overall'].item()

            if self.is_master:
                tbar.set_description(
                    f'Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}'
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
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                    break
                
            wandb.log({
                'early_stopping/counter': self.early_stopping_counter,
                'early_stopping/patience': self.early_stopping_patience,
                'epoch': epoch,
            })
            
            current_val_loss = val_loss / len(self.val_data)
            if current_val_loss < self.best_val_loss:
                logger.info(
                    f'New best model with validation loss: {current_val_loss:.4f} at epoch {epoch}'
                )
                self.best_val_loss = current_val_loss
                save_checkpoint(
                    self.net,
                    self.cfg.data_paths.CHECKPOINTS_PATH,
                    prefix='loss',
                    best=True,
                    multi_gpu=self.cfg.multi_gpu,
                )
               

    def iterative_evaluation(self, epoch):
        validation_set = self.val_dataset
        
        evaluation_clicker_config = {}
    
        if self.clicker_config['mode'] == 'distributed':
            evaluation_clicker_config['mode'] = 'avg' # Deterministic
        elif self.clicker_config['mode'] == 'distributed_only_pos':
            evaluation_clicker_config['mode'] = 'avg_only_pos' # Deterministic
        else:
            evaluation_clicker_config['mode'] = self.clicker_config['mode']
            
        evaluation_clicker_config['size_range_modifier'] = 0 # Deterministic
        
        zoom_in_params = dict()
        zoom_in_params['recompute_click_size_on_zoom'] = evaluation_clicker_config['mode'] != 'locked'
        logger.info(f'Zoom in params: {zoom_in_params}')
        
        predictor_params = dict()
        logger.info(f'Predictor params: {predictor_params}')
        
        predictor = get_predictor(
            self.cfg.model_type,
            self.net,
            self.device,
            zoom_in_params,
            predictor_params,
        )
        
        with torch.no_grad():
            avg_ious, noc, nof, iou_error, _ = iterative_evaluate_dataset(
                validation_set, predictor, logger, evaluation_clicker_config
            )
            
            logger.info('-----------------------------------------------------')
            logger.info(f'AVERAGE IOUS AFTER {epoch} EPOCHS: {avg_ious}')
            logger.info(f'NoC AFTER {epoch} EPOCHS: {noc}')
            logger.info(f'NoF AFTER {epoch} EPOCHS: {nof}')
            logger.info(f'IoU ERROR AFTER {epoch} EPOCHS: {iou_error}')
            logger.info('-----------------------------------------------------')
            
            sample_count = len(validation_set)
            wandb.log(
                {
                    'iterative_evaluation_val/Iterative IoU Error': iou_error,
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

        if iou_error < self.best_iterative_error:
            self.best_iterative_error = iou_error
            logger.info(
                f'New best iterative eval model with error: {iou_error} at epoch {epoch}'
                )
            save_checkpoint(
                self.net,
                self.cfg.data_paths.CHECKPOINTS_PATH,
                prefix='iterative',
                best=True,
                multi_gpu=self.cfg.multi_gpu,
            )
    
    def batch_forward(self, batch_data, validation=False):
        raise Exception('Not implemented')

    def add_loss(
        self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs
    ):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    @property
    def is_master(self):
        return self.cfg.local_rank == 0
