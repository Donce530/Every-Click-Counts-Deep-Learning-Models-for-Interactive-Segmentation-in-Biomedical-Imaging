from pathlib import Path

import numpy as np
import os
import torch
from torch.utils.data import Dataset

from isegm.ritm.data.base import ISDataset
from isegm.ritm.data.sample import DSample

from isegm.data.preprocess import Preprocessor


class LidcDataset(ISDataset):
    def __init__(
        self,
        dataset_path,
        preprocessor: Preprocessor = None,
        images_dir_name="slices",
        masks_dir_name="slice_annotations",
        initial_masks_dir="initial_masks/active_contours",
        **kwargs,
    ):
        super(LidcDataset, self).__init__(**kwargs)

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
        initial_mask_path = str(self._initial_masks_path / image_name)

        image = np.load(image_path).astype(np.float32)

        instances_mask = np.load(mask_path).astype(int)
        instances_mask[instances_mask > 0] = 1

        initial_mask = np.load(initial_mask_path).astype(int)
        initial_mask[instances_mask > 0] = 1

        if self.preprocessor:
            image = self.preprocessor.preprocess(image)

        return DSample(
            image,
            instances_mask,
            objects_ids=[1],
            ignore_ids=[-1],
            sample_id=index,
            init_mask=initial_mask,
        )


class Lidc2dDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.slice_path = os.path.join(data_dir, "slices")
        self.mask_path = os.path.join(data_dir, "slice_annotations")
        self.transform = transform
        self.file_names = os.listdir(self.slice_path)

        assert os.listdir(self.slice_path) == os.listdir(
            self.mask_path
        ), "Mismatch in number of files."

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]

        slice = np.load(os.path.join(self.slice_path, file_name)).astype(np.float32)
        mask = np.load(os.path.join(self.mask_path, file_name)).astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image=slice, mask=mask)
            slice, mask = transformed["image"], transformed["mask"]

        return torch.from_numpy(slice), torch.from_numpy(mask)
