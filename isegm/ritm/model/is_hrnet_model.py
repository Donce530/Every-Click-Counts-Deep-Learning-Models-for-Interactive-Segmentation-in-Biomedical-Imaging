import torch.nn as nn

from isegm.ritm.utils.serialization import serialize
from isegm.ritm.model.modifiers import LRMult
# from .is_model_single_channel import ISModel
from .is_model import ISModel
from .modeling.hrnet_ocr import HighResolutionNet


class HRNetModel(ISModel):
    @serialize
    def __init__(
        self,
        width=48,
        ocr_width=256,
        small=False,
        backbone_lr_mult=0.1,
        norm_layer=nn.BatchNorm2d,
        **kwargs,
    ):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = HighResolutionNet(
            width=width,
            ocr_width=ocr_width,
            small=small,
            num_classes=1,
            # num_channels=1,
            norm_layer=norm_layer,
        )
        self.set_backbone_lr_multiplier(backbone_lr_mult)
            
    def set_backbone_lr_multiplier(self, multiplier=0.1):
        self.feature_extractor.apply(LRMult(multiplier))
        if self.feature_extractor.ocr_width > 0:
            self.feature_extractor.ocr_distri_head.apply(LRMult(1.0))
            self.feature_extractor.ocr_gather_head.apply(LRMult(1.0))
            self.feature_extractor.conv3x3_ocr.apply(LRMult(1.0))

    def backbone_forward(self, image, coord_features=None):
        net_outputs = self.feature_extractor(image, coord_features)

        return {"instances": net_outputs[0], "instances_aux": net_outputs[1], "pre_stage_features": net_outputs[2]}