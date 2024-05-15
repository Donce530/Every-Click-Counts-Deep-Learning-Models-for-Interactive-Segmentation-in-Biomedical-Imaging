import numpy as np
import cv2
from isegm.classical_approaches.snake_fill import SnakeMasker
from isegm.data.datasets import Lidc2dDataset
from torch.utils.data import DataLoader
from albumentations import Compose, Normalize
from isegm.inference.utils import get_iou
import os


def do():
    print(f'Starting:')

    # Define paths
    base_path = '/gpfs/space/projects/PerkinElmer/donatasv_experiments/datasets/processed_datasets/LIDC-2D'
    # datasets = ['train', 'val', 'test']
    datasets = ['val']

    # Define transformations for normalization
    transformations = Compose([Normalize(mean=0, std=1)])

    # Specific parameter for SnakeMasker
    circle_size_param = 8
    masker = SnakeMasker(circle_size=circle_size_param)

    ious = []

    for dataset in datasets:
        print(f"Processing {dataset} dataset...")
        dataset_path = os.path.join(base_path, dataset)
        output_path = os.path.join(
            base_path, dataset, 'initial_masks', 'active_contours'
        )

        transformations = Compose([Normalize(mean=0, std=1)])
        dataset = Lidc2dDataset(data_dir=dataset_path, transform=transformations)

        train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for i, (image, gt_mask) in enumerate(train_loader):
            gt_mask = gt_mask.squeeze().numpy()
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                image.squeeze().numpy().astype(np.uint8)
            )
            coords = [
                (int(centroids[label][1]), int(centroids[label][0]))
                for label in range(1, num_labels)
            ]
            labels = masker.predict(image.squeeze().numpy(), coords)

            iou = get_iou(gt_mask, labels)
            ious.append(iou)

            print(f'{i}: {iou}')

            # file_path = os.path.join(output_path, dataset.file_names[i])
            # np.save(file_path, labels.astype(bool))
            # print(f'Saved file {i}: {file_path}')

        print(np.array(iou).mean())

    print('Done')


if __name__ == "__main__":
    do()
