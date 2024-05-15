from skimage.segmentation import flood
import numpy as np


class FloodMasker:
    def __init__(self, tolerance=0.5):
        self.tolerance = tolerance

    def predict(self, image, coordinate_list):
        mask = np.zeros(image.shape, dtype=bool)
        for coordinates in coordinate_list:
            coordinates = (coordinates[1], coordinates[0]) # reverse to x, y from row, col
            mask += flood(image, coordinates, tolerance=self.tolerance)

        return mask
