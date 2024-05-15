from isegm.data.aligned_augmentation import ProbabilisticAlignedAugmentator, AlignedAugmentator
from albumentations import Compose, RandomCrop, PadIfNeeded, HorizontalFlip, IAAAffine, ElasticTransform
from isegm.data.transforms import UniformRandomResize
from isegm.utils.log import logger
from isegm.inference import utils
import cv2
import numpy as np

class AugmentationsProvider:
    def __init__(self):
        self.augmentation_map = {
            'target_crop': self._get_target_crop_augmentator,
            'ritm_standard': self._get_ritm_standard_augmentator,
            'focalclick_standard': self._get_focalclick_standard_augmentator,
            'ct_custom': self._get_ct_custom_augmentator,
            'none': self._get_empty_augmentator,
        }
        
    def get_augmentator(self, cfg, model_cfg):
        if cfg.augmentation_type in self.augmentation_map:
            augmentation_fn = self.augmentation_map[cfg.augmentation_type]
            return augmentation_fn(cfg, model_cfg)
        else:
            raise Exception(f'Unknown augmentation type: {cfg.augmentation_type}')
    
    def _get_empty_augmentator(self, cfg, model_cfg):
        return Compose([], p=1.0)
        
    def _get_target_crop_augmentator(self, cfg, model_cfg):
        if cfg.augmentation_type == 'target_crop':
            target_crop_size = (
                cfg.target_crop_augmentation.crop_size,
                cfg.target_crop_augmentation.crop_size,
            )
            
            bounds = self._calcuclate_target_crop_bounds(cfg)
            
            augmentator = ProbabilisticAlignedAugmentator(
                ratio=[
                    bounds[0], # lower bound
                    bounds[1], # upper bound
                ],
                target_size=target_crop_size,
                crop_probability=cfg.target_crop_augmentation.crop_probability,
                flip=True,
                distribution='Gaussian',
                gs_center=1,
            )
            logger.info(f'Applied target crop augmentation for train dataset. Input size: {target_crop_size}')
            return augmentator

    def _calcuclate_target_crop_bounds(self, cfg):
        validation_dataset = utils.get_dataset(cfg.dataset.val, cfg.data_paths)
        def get_box_corners(mask):
            mask = mask.astype(np.uint8)
            mask[mask > 0] = 1
            mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            return x, y, x + w, y + h
            
        img_height, img_width = validation_dataset.get_sample(0).image.shape[:2]
        widths = []
        heights = []
        for i in range(len(validation_dataset)):
            sample = validation_dataset.get_sample(i)
            mask = sample.gt_mask
            x1, y1, x2, y2 = get_box_corners(mask)
            widths.append(x2 - x1)
            heights.append(y2 - y1)

        mean_width = np.mean(widths)
        mean_height = np.mean(heights)
        width_percentage = mean_width / img_width
        height_percentage = mean_height / img_height
        average_ratio = (width_percentage + height_percentage) / 2
        bounds = np.round(average_ratio * 0.5, 3), np.round(average_ratio * 1.5, 3)
        bounds = (bounds[0] * 4, bounds[1] * 4)
        
        logger.info(f'Target crop bounds for {cfg.dataset.val} dataset: {bounds}')
        
        return bounds
        
    def _get_ritm_standard_augmentator(self, cfg, model_cfg):
        train_augmentator = Compose(
            [
                UniformRandomResize(scale_range=(0.75, 1.40)),
                HorizontalFlip(),
                PadIfNeeded(
                    min_height=model_cfg.crop_size[0], min_width=model_cfg.crop_size[1], border_mode=0
                ),
                RandomCrop(*model_cfg.crop_size),
            ],
            p=1.0,
        )
        logger.info(f'Applied RITM standard augmentations for train dataset. Input size: {model_cfg.crop_size}')
        return train_augmentator
    
    def _get_focalclick_standard_augmentator(self, cfg, model_cfg):
        train_augmentator = AlignedAugmentator(ratio=[0.5,1.4], target_size=model_cfg.crop_size ,flip=True, 
                                            distribution='Gaussian', gs_center=1, 
                                            color_augmentator=None) # No color augmentations
        logger.info(f'Applied FocalClick standard augmentations for train dataset. Input size: {model_cfg.crop_size}')
        return train_augmentator
    
    def _get_ct_custom_augmentator(self, cfg, model_cfg):
        train_augmentator = Compose(
            [
                UniformRandomResize(scale_range=(0.75, 1.40)),
                HorizontalFlip(),
                # IAAAffine(rotate=(-45, 45), mode='constant'),
                ElasticTransform(alpha=1, sigma=25, alpha_affine=25),
                PadIfNeeded(
                    min_height=model_cfg.crop_size[0], min_width=model_cfg.crop_size[1], border_mode=0
                ),
                RandomCrop(*model_cfg.crop_size),
            ],
            p=1.0,
        )
        logger.info(f'Applied CT custom augmentations for train dataset. Input size: {model_cfg.crop_size}')
        return train_augmentator