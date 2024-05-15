from pathlib import Path
import cv2

import numpy as np
import os
import torch
from torch.utils.data import Dataset

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class BratsDataset(ISDataset):
    def __init__(
        self,
        dataset_path,
        num_channels=None,
        merge_masks=[2, 4],
        images_dir_name="slices",
        masks_dir_name="slice_annotations",
        **kwargs
    ):
        super(BratsDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob("*.*"))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob("*.*")}
        self.merge_masks = merge_masks
        self.num_channels = num_channels

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split(".")[0]])

        image = np.load(image_path).astype(np.float32)

        if self.num_channels is not None:
            image = image[:, :, : self.num_channels]

        mask = np.load(mask_path).astype(int)

        if len(self.merge_masks) > 0:
            mask = np.isin(mask, self.merge_masks).astype(int)

        return DSample(
            image,
            mask,
            objects_ids=[1],
            ignore_ids=[-1],
            sample_id=index,
        )


class BratsSimpleClickDataset(ISDataset):
    def __init__(
        self,
        dataset_path,
        images_dir_name="image",
        masks_dir_name="annotation",
        **kwargs
    ):
        super(BratsSimpleClickDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob("*.*"))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob("*.*")}

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split(".")[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        instances_mask[instances_mask > 0] = 1

        return DSample(
            image,
            instances_mask,
            objects_ids=[1],
            ignore_ids=[-1],
            sample_id=index,
        )


class Brats2dDataset(Dataset):
    def __init__(self, data_dir, merge_masks=[2, 4], transform=None):
        self.slice_path = os.path.join(data_dir, "slices")
        self.mask_path = os.path.join(data_dir, "slice_annotations")
        self.merge_masks = merge_masks
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

        if len(self.merge_masks) > 0:
            mask = np.isin(mask, self.merge_masks).astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image=slice, mask=mask)
            slice, mask = transformed["image"], transformed["mask"]

        return torch.from_numpy(slice), torch.from_numpy(mask)
