import numpy as np
import os
import pandas as pd
from isegm.data.datasets import Lidc2dDataset
from torch.utils.data import DataLoader
from albumentations import Compose, Normalize
from isegm.inference.utils import get_iou, get_f1_score
from isegm.classical_approaches.snake_fill import SnakeMasker
from isegm.classical_approaches.flood_fill import FloodMasker
from isegm.data.preprocess import Preprocessor
import cv2
import wandb


def run():
    # Define hardcoded constants
    base_path = '/gpfs/space/projects/PerkinElmer/donatasv_experiments/datasets/processed_datasets/LIDC-2D'
    train_dataset_name = 'train'
    val_dataset_name = 'val'
    test_dataset_name = 'test'
    batch_size = 1
    threshold_range = np.arange(0.05, 2.001, 0.025)

    # Define transformations for normalization
    transformations = Compose([Normalize(mean=0, std=1)])
    preprocessor = Preprocessor()
    preprocessor.normalize = False
    preprocessor.windowing = True
    preprocessor.num_input_channels = 1
    preprocessor.window_min = -900
    preprocessor.window_max = 600

    # Prepare the train dataset
    train_dataset_path = os.path.join(base_path, train_dataset_name)
    train_dataset = Lidc2dDataset(
        data_dir=train_dataset_path, transform=transformations, preprocessor=preprocessor
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Prepare the validation dataset
    val_dataset_path = os.path.join(base_path, val_dataset_name)
    val_dataset = Lidc2dDataset(data_dir=val_dataset_path, transform=transformations, preprocessor=preprocessor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    
    test_dataset_path = os.path.join(base_path, test_dataset_name)
    test_dataset = Lidc2dDataset(data_dir=test_dataset_path, transform=transformations, preprocessor=preprocessor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize WandB
    wandb.init(project='classical_methods', name='snake-param-search')
    
    def compute_scores(treshold, data_loader):
        masker = FloodMasker(tolerance=treshold)
        iou_scores = []
        f1_scores = []
        for i, (image, gt_mask) in enumerate(data_loader):
            print(f'Testing image: {i}, Threshold: {treshold}, Progress: {i/len(data_loader) * 100:.2f}%')
            image_np = image.squeeze().squeeze().numpy()
            gt_mask = gt_mask.squeeze().numpy()
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                gt_mask.astype(np.uint8)
            )
            coords = [
                (int(centroids[label][0]), int(centroids[label][1]))
                for label in range(1, num_labels)
            ][0:1] # only first centroid
            pred_mask = masker.predict(image_np, coords).astype(bool)
            iou_scores.append(get_iou(gt_mask, pred_mask))
            f1_scores.append(get_f1_score(gt_mask, pred_mask))
        return np.mean(iou_scores), np.mean(f1_scores)


    # Perform parameter search for best circle size
    
    best_treshold = None
    best_mean_iou = 0
    best_mean_f1 = 0
    results_df = pd.DataFrame(columns=['treshold', 'mean_iou', 'mean_f1'])
    for treshold in threshold_range:
        print(f'Computing scores for treshold: {treshold}')
        mean_iou, mean_f1 = compute_scores(treshold, train_loader)
        wandb.log(
            {'treshold': treshold, 'mean_iou': mean_iou, 'mean_f1': mean_f1}
        )
        if mean_iou > best_mean_iou:
            best_treshold = treshold
            best_mean_iou = mean_iou
            best_mean_f1 = mean_f1
        results_df = results_df.append({'treshold': treshold, 'mean_iou': mean_iou, 'mean_f1': mean_f1}, ignore_index=True)
    
    results_df.to_csv('classical_search_flood_lidc.csv')

    print(
        f'Best treshold size: {best_treshold}, Mean IoU: {best_mean_iou:.4f}, Mean F1: {best_mean_f1:.4f}'
    )

    # Evaluate on validation set with the best circle size
    val_mean_iou, val_mean_f1 = compute_scores(best_treshold, val_loader)
    print(
        f'Validation Mean IoU: {val_mean_iou:.4f}, Validation Mean F1: {val_mean_f1:.4f}'
    )
    
    test_mean_iou, test_mean_f1 = compute_scores(best_treshold, test_loader)
    print(
        f'Test Mean IoU: {test_mean_iou:.4f}, Test Mean F1: {test_mean_f1:.4f}'
    )

    # Log results to WandB
    wandb.log(
        {
            'best_threshold': best_treshold,
            'validation_mean_iou': val_mean_iou,
            'validation_mean_f1': val_mean_f1,
        }
    )
    wandb.finish()



if __name__ == "__main__":
    run()
