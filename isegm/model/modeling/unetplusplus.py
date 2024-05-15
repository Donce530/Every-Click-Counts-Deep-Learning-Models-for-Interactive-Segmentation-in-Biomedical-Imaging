import segmentation_models_pytorch as smp
import torch

class UnetPlusPlus(torch.nn.Module):
    def __init__(self, encoder_weights='imagenet', with_prev_mask=False):
        super().__init__()
        # Initialize U-Net with ResNet50 backbone
        in_channels = 6 if with_prev_mask else 5
        
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights, # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )

    def forward(self, image, points):
        # Concatenate image and points along the channel dimension
        x = torch.cat([image, points], dim=1)
        # Forward pass through the U-Net++ model
        return self.model(x)