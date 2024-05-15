from pathlib import Path

import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from isegm.data.preprocess import Preprocessor
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class MdPancDataset(ISDataset):
    def __init__(
        self,
        dataset_path,
        preprocessor: Preprocessor = None,
        images_dir_name="slices",
        masks_dir_name="slice_annotations",
        initial_masks_dir="initial_masks/active_contours",
        **kwargs,
    ):
        super(MdPancDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name
        self._initial_masks_path = self.dataset_path / initial_masks_dir

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob("*.*"))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob("*.*")}
        self.preprocessor = preprocessor

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split(".")[0]])

        image = np.load(image_path).astype(np.float32)
        instances_mask = np.load(mask_path).astype(int)

        if self.preprocessor:
            image = self.preprocessor.preprocess(image)

        return DSample(
            image,
            instances_mask,
            objects_ids=[2], # 1 - kidney, 2 - tumour, 3 - cyst
            ignore_ids=[-1],
            sample_id=index,
        )
        
    def get_sample_paths(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split(".")[0]])
        return image_path, mask_path