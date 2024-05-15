from isegm.model.is_model import ISModel
from torch import nn
import torch
from isegm.model.ops import DistMaps
from isegm.utils.serialization import serialize
from .modeling.unetplusplus import UnetPlusPlus

class UnetPlusPlusModelExplainable(ISModel):
    @serialize
    def __init__(self, norm_layer=nn.BatchNorm2d, dist_map_mode='disk', dist_map_radius=5, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        with_prev_mask = kwargs.get('with_prev_mask', False)
        self.feature_extractor = UnetPlusPlus(with_prev_mask=with_prev_mask)
        
        self.dist_maps_base = DistMaps(norm_radius=dist_map_radius, spatial_scale=1.0,
                                      cpu_mode=False, mode=dist_map_mode)
                                    
    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps_base(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features
    
    def backbone_forward(self, image, coord_features=None):
        output = self.feature_extractor(image, coord_features)
        return {'instances': output}

    def forward(self, image, points):
        
        info = {}
        
        image, prev_mask = self.prepare_input(image)
        
        info['points'] = points
        info['prepared_input'] = {'image': image, 'prev_mask': prev_mask}
        
        coord_features = self.get_coord_features(image, prev_mask, points)
        
        click_map = coord_features[:, 1:, :, :]
        info['coord_features'] = coord_features
        info['click_map'] = click_map

        outputs = self.backbone_forward(image, coord_features)
        
        info['outputs'] = outputs

        return outputs, info