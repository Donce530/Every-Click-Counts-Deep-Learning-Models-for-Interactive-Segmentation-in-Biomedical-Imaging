import torch.nn as nn
import segmentation_models_pytorch as smp

from isegm.ritm.utils.serialization import serialize
from isegm.ritm.model.modifiers import LRMult
from .is_model_single_channel import ISModel

class UNetPlusPlusModel(ISModel):
    @serialize
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_depth=5,
        encoder_weights="imagenet",
        decoder_use_batchnorm=True,
        decoder_attention_type=None,  # Attention mechanism (e.g., 'scse')
        backbone_lr_mult=0.1,
        norm_layer=nn.BatchNorm2d,
        **kwargs,
    ):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_attention_type=decoder_attention_type,
            in_channels=1,
            classes=1,
            activation=None
        )
        self.set_backbone_lr_multiplier(backbone_lr_mult)
        
    def set_backbone_lr_multiplier(self, multiplier=0.1):
        self.feature_extractor.apply(LRMult(multiplier))

    def backbone_forward(self, image, coord_features=None):
        mask = self.feature_extractor(image)
        return {"instances": mask, "instances_aux": mask}
