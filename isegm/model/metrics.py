import torch
import numpy as np
import wandb

from isegm.utils import misc


class TrainMetric(object):
    def __init__(self, pred_outputs, gt_outputs):
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_epoch_value(self):
        raise NotImplementedError

    def reset_epoch_stats(self):
        raise NotImplementedError

    def log_states(self, epoch, tag_prefix=None):
        pass

    @property
    def name(self):
        return type(self).__name__


class AdaptiveIoU(TrainMetric):
    def __init__(
        self,
        init_thresh=0.4,
        # init_thresh=0.5, tested once (maybe makes sense for pretrained models)
        thresh_step=0.025,
        thresh_beta=0.99,
        iou_beta=0.9,
        ignore_label=-1,
        from_logits=True,
        pred_output='instances',
        gt_output='instances',
    ):
        super().__init__(pred_outputs=(pred_output,), gt_outputs=(gt_output,))
        self._ignore_label = ignore_label
        self._from_logits = from_logits
        self._iou_thresh = init_thresh
        self._thresh_step = thresh_step
        self._thresh_beta = thresh_beta
        self._iou_beta = iou_beta
        self._ema_iou = 0.0
        self._epoch_iou_sum = 0.0
        self._epoch_batch_count = 0

    def update(self, pred, gt):
        gt_mask = gt > 0.5
        if self._from_logits:
            pred = torch.sigmoid(pred)

        gt_mask_area = torch.sum(gt_mask, dim=(1, 2)).detach().cpu().numpy()
        if np.all(gt_mask_area == 0):
            return

        ignore_mask = gt == self._ignore_label
        max_iou = _compute_iou(pred > self._iou_thresh, gt_mask, ignore_mask).mean()
        best_thresh = self._iou_thresh
        for t in [best_thresh - self._thresh_step, best_thresh + self._thresh_step]:
            temp_iou = _compute_iou(pred > t, gt_mask, ignore_mask).mean()
            if temp_iou > max_iou:
                max_iou = temp_iou
                best_thresh = t

        self._iou_thresh = (
            self._thresh_beta * self._iou_thresh + (1 - self._thresh_beta) * best_thresh
        )
        self._ema_iou = self._iou_beta * self._ema_iou + (1 - self._iou_beta) * max_iou
        self._epoch_iou_sum += max_iou
        self._epoch_batch_count += 1

    def get_epoch_value(self):
        if self._epoch_batch_count > 0:
            return self._epoch_iou_sum / self._epoch_batch_count
        else:
            return 0.0

    def reset_epoch_stats(self):
        self._epoch_iou_sum = 0.0
        self._epoch_batch_count = 0

    def log_states(self, epoch, tag_prefix=None):
        tag_prefix = f'{tag_prefix}_' if tag_prefix is not None else ''
        wandb.log(
            {
                f"{tag_prefix}metrics/{self.name}": self.get_epoch_value(),
                f"{tag_prefix}metrics/{self.name}_treshold": self._iou_thresh,
                "epoch": epoch,
            }
        )

    @property
    def iou_thresh(self):
        return self._iou_thresh


def _compute_iou(pred_mask, gt_mask, ignore_mask=None, keep_ignore=False):
    if ignore_mask is not None:
        pred_mask = torch.where(ignore_mask, torch.zeros_like(pred_mask), pred_mask)

    reduction_dims = misc.get_dims_with_exclusion(gt_mask.dim(), 0)
    union = (
        torch.mean((pred_mask | gt_mask).float(), dim=reduction_dims)
        .detach()
        .cpu()
        .numpy()
    )
    intersection = (
        torch.mean((pred_mask & gt_mask).float(), dim=reduction_dims)
        .detach()
        .cpu()
        .numpy()
    )
    nonzero = union > 0

    iou = intersection[nonzero] / union[nonzero]
    if not keep_ignore:
        return iou
    else:
        result = np.full_like(intersection, -1)
        result[nonzero] = iou
        return result


class IoU(TrainMetric):
    def __init__(self, ignore_label=-1, pred_output='instances', gt_output='instances'):
        super().__init__(pred_outputs=(pred_output,), gt_outputs=(gt_output,))
        self._ignore_label = ignore_label
        self._iou_sum = 0.0
        self._batch_count = 0

    def update(self, pred, gt):
        gt_mask = gt > 0.5  # Assuming binary masks
        intersection = torch.sum(pred[gt_mask] > 0.5).item()
        union = torch.sum(pred > 0.5).item() + torch.sum(gt_mask).item() - intersection

        iou = intersection / union if union > 0 else 0.0
        self._iou_sum += iou
        self._batch_count += 1

    def get_epoch_value(self):
        if self._batch_count > 0:
            return self._iou_sum / self._batch_count
        else:
            return 0.0

    def reset_epoch_stats(self):
        self._iou_sum = 0.0
        self._batch_count = 0

    def log_states(self, epoch, tag_prefix=None):
        tag_prefix = f'{tag_prefix}_' if tag_prefix is not None else ''
        wandb.log(
            {
                f"{tag_prefix}metrics/{self.name}": self.get_epoch_value(),
                "epoch": epoch,
            }
        )


class F1Score(TrainMetric):
    def __init__(self, ignore_label=-1, pred_output='instances', gt_output='instances'):
        super().__init__(pred_outputs=(pred_output,), gt_outputs=(gt_output,))
        self._ignore_label = ignore_label
        self._precision_sum = 0.0
        self._recall_sum = 0.0
        self._batch_count = 0

    def update(self, pred, gt):
        pred_mask = (pred > 0.5).float()
        gt_mask = (gt > 0.5).float()
        true_positive = torch.sum(pred_mask * gt_mask).item()
        false_positive = torch.sum(pred_mask * (1 - gt_mask)).item()
        false_negative = torch.sum((1 - pred_mask) * gt_mask).item()

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        self._precision_sum += precision
        self._recall_sum += recall
        self._batch_count += 1

    def get_epoch_value(self):
        if self._batch_count > 0:
            precision_avg = self._precision_sum / self._batch_count
            recall_avg = self._recall_sum / self._batch_count
            return (
                2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)
                if precision_avg + recall_avg > 0
                else 0.0
            )
        else:
            return 0.0

    def reset_epoch_stats(self):
        self._precision_sum = 0.0
        self._recall_sum = 0.0
        self._batch_count = 0

    def log_states(self, epoch, tag_prefix=None):
        tag_prefix = f'{tag_prefix}_' if tag_prefix is not None else ''
        wandb.log(
            {
                f"{tag_prefix}metrics/{self.name}": self.get_epoch_value(),
                "epoch": epoch,
            }
        )
