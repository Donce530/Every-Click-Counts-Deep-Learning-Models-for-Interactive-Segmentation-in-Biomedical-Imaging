from pathlib import Path

import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from isegm.data.preprocess import Preprocessor
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class CombinedDataset(ISDataset):
    def __init__(
        self,
        dataset_path,
        preprocessor: Preprocessor = None,
        images_dir_name="slices",
        masks_dir_name="slice_annotations",
        subdataset_filter=None,
        **kwargs,
    ):
        super(CombinedDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name
        self.subdataset_filter = subdataset_filter  
        self.preprocessor = None
        
        self.dataset_object_ids = {
            'KITS23-2D-TUMOURS': [2],
            'LIDC-2D': None,
            'LITS_2D': [2],
            'MD_PANC_2D': [2],
            'KITS23-2D-TUMOURS-FULL': [2],
            'LIDC-2D-FULL': None,
            'LITS_2D_FULL': [2],
            'MD_PANC_2D_FULL': [2],
        }

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob("*.*"))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob("*.*")}
        
        if self.subdataset_filter:
            self.dataset_samples = [x for x in self.dataset_samples if x.split("__")[0] in self.subdataset_filter]
        
        self.preprocessor = preprocessor

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split(".")[0]])

        image = np.load(image_path).astype(np.float32)
        instances_mask = np.load(mask_path).astype(int)
        
        object_ids = self.dataset_object_ids[image_name.split("__")[0]]
        if object_ids is not None:
            instances_mask = np.isin(instances_mask, object_ids)
        else: # Consider everything
            instances_mask = instances_mask > 0
        instances_mask = instances_mask.astype(int)

        if self.preprocessor:
            image = self.preprocessor.preprocess(image)

        return DSample(
            image,
            instances_mask,
            objects_ids=[1],
            ignore_ids=[-1],
            sample_id=index,
        )
        
    def get_sample_paths(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split(".")[0]])
        return image_path, mask_path