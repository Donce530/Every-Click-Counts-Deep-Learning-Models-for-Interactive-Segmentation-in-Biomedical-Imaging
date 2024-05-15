import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.draw import polygon


class SnakeMasker:
    def __init__(self, circle_size):
        self.circle_size = circle_size

    def predict(self, image, coordinate_list):
        mask = np.zeros_like(image, dtype=bool)

        for x, y in coordinate_list:
            s = np.linspace(0, 2 * np.pi, 400)
            r = y + self.circle_size * np.sin(s)  # r corresponds to y
            c = x + self.circle_size * np.cos(s)  # c corresponds to x
            init = np.array([r, c]).T

            snake = active_contour(
                gaussian(image, 3, preserve_range=False),
                init,
                alpha=0.015,
                beta=10,
                gamma=0.001,
                coordinates='rc',
            )

            rr, cc = polygon(snake[:, 0], snake[:, 1])
            mask[rr, cc] = 1

        return mask
