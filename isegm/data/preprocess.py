import numpy as np
import cv2

from PIL import Image, ImageEnhance, ImageFilter


class Preprocessor:
    def __init__(self, configuration=None) -> None:
        if configuration == None:
            self.enhancements = []
            self.normalize = True
            self.windowing = False
            self.convert_to_rgb = False
            self.num_input_channels = 3
        else:
            self.enhancements = (
                [
                    (key, value)
                    for key, value in configuration['enhancements'].items()
                    if value >= 0 and value != None
                ]
                if configuration['enhancements'] != None
                else []
            )

            self.normalize = (
                configuration['normalize']
                if 'normalize' in configuration.keys()
                else True
            )

            window_params = configuration['windowing']
            self.windowing = window_params['enabled']
            if self.windowing:
                self.window_min = window_params['min']
                self.window_max = window_params['max']

            self.convert_to_rgb = configuration['convert_to_rgb']
        
            # backwards compatibility
            if 'num_input_channels' in configuration:
                self.num_input_channels = configuration['num_input_channels']
            else:
                self.num_input_channels = 3

    def preprocess(self, image):
        if self.windowing:
            image = self._window_image(image)

        if self.normalize:
            max_val = np.max(image)
            min_val = np.min(image)

            image = image - min_val
            image = image / (max_val - min_val) * 255

        if self.convert_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.expand_dims(image, axis=2)
            image = np.repeat(image, self.num_input_channels, axis=2)

        if len(self.enhancements) > 0:
            image = self._enhance_image(image, self.enhancements)

        return image.astype(np.float32)

    def _window_image(self, image):
        return np.clip(image, self.window_min, self.window_max)

    def _enhance_image(self, img, transforms=[]):
        transform_types = {
            'contrast': lambda img, factor: ImageEnhance.Contrast(img).enhance(factor),
            'brightness': lambda img, factor: ImageEnhance.Brightness(img).enhance(
                factor
            ),
            'sharpness': lambda img, factor: ImageEnhance.Sharpness(img).enhance(
                factor
            ),
            'gaussian_blur': lambda img, factor: img.filter(
                ImageFilter.GaussianBlur(radius=factor)
            ),
        }
        img = Image.fromarray(img.astype(np.uint8))

        for transform_type, transform_factor in transforms:
            img = transform_types[transform_type](img, transform_factor)

        return np.array(img)
