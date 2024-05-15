import segmentation_models_pytorch as smp
import lightning.pytorch as pl
import torch.optim as optim
import torch
from isegm.ritm.model.losses import FocalLossPT
from torchmetrics.classification import BinaryJaccardIndex, F1Score


class UnetLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=4,
            classes=1,
        )
        self.loss_fn = FocalLossPT()
        self.iou_metric = BinaryJaccardIndex()
        self.f1_metric = F1Score('binary')

    def training_step(self, batch, batch_idx):
        images, masks = batch

        images = torch.swapaxes(images, 1, 3)

        prob_map = self.model(images)
        prob_map = torch.squeeze(prob_map, dim=1)

        loss = self.loss_fn(prob_map, masks)
        self.log("train_loss", loss)

        predicted_masks = (prob_map > 0.5).int()
        real_masks = (masks).int()

        iou = self.iou_metric(predicted_masks, real_masks)
        self.log("train_iou", iou)

        f1 = self.f1_metric(predicted_masks, real_masks)
        self.log("train_f1", f1)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        images = torch.swapaxes(images, 1, 3)

        prob_map = self.model(images)
        prob_map = torch.squeeze(prob_map, dim=1)

        loss = self.loss_fn(prob_map, masks)
        self.log("val_loss", loss)

        predicted_masks = (prob_map > 0.5).int()
        real_masks = (masks).int()

        iou = self.iou_metric(predicted_masks, real_masks)
        self.log("val_iou", iou)

        f1 = self.f1_metric(predicted_masks, real_masks)
        self.log("val_f1", f1)
