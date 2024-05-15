# from isegm.data.datasets import Lidc2dDataset
# from albumentations import Compose, Normalize
# from torch.utils.data import DataLoader
# from classical_approaches.flood_fill import FloodMasker
# from classical_approaches.snake_fill import SnakeMasker
# from sklearn.metrics import f1_score
# from isegm.inference.utils import get_iou
# import wandb
# import numpy as np
# from tqdm import tqdm
# import cv2

# from omegaconf import DictConfig, OmegaConf
# import hydra


# @hydra.main(version_base=None, config_path="./conf", config_name="config")
# def run(cfg: DictConfig):
#     print(OmegaConf.to_yaml(cfg))

#     match cfg.dataset.name:
#         case 'LIDC-2D':
#             transformations = Compose([Normalize(mean=0, std=1)])
#             dataset = Lidc2dDataset(
#                 data_dir=cfg.dataset.train_path, transform=transformations
#             )
#             train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
#             val_dataset = Lidc2dDataset(
#                 data_dir=cfg.dataset.val_path, transform=transformations
#             )
#             val_loader = DataLoader(
#                 val_dataset, batch_size=cfg.batch_size, shuffle=False
#             )
#         case _:
#             raise NotImplementedError(f'Dataset {cfg.dataset.name} not implemented.')

#     match cfg.classical_method.name:
#         case 'flood':
#             model_class = FloodMasker
#             parameter_range = np.arange(
#                 cfg.classical_method.tolerance.min,
#                 cfg.classical_method.tolerance.max,
#                 cfg.classical_method.tolerance.step,
#             )
#             parameter_to_test = 'tolerance'
#         case 'snakes':
#             model_class = SnakeMasker
#             # parameter_range = np.arange(
#             #     cfg.classical_method.circle_sizes.min,
#             #     cfg.classical_method.circle_sizes.max,
#             #     cfg.classical_method.circle_sizes.step,
#             # )
#             parameter_range = np.arange(7, 10, 1)
#             parameter_to_test = 'circle_size'
#         case _:
#             raise NotImplementedError(
#                 f'Classical method {cfg.classical_method} not implemented.'
#             )

#     wandb.init(
#         # Set the project where this run will be logged
#         project='classical_methods',
#         # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#         name=f'{cfg.classical_method.name}-{cfg.dataset.name}-param_search',
#         # Track hyperparameters and run metadata
#         config=cfg,
#     )

#     scores = np.zeros(len(parameter_range))
#     truths = []
#     coords = []

#     for image, gt_mask in train_loader:
#         gt_mask = gt_mask.squeeze().numpy()
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
#             gt_mask.astype(np.uint8)
#         )
#         coords.append(
#             [
#                 (int(centroids[label][1]), int(centroids[label][0]))
#                 for label in range(1, num_labels)
#             ]
#         )
#         truths.append(gt_mask)

#     truths = np.array(truths)

#     for i, parameter_value in tqdm(enumerate(parameter_range)):
#         predictions = []
#         masker = model_class(parameter_value)

#         for j, (image, gt_mask) in enumerate(train_loader):
#             labels = masker.predict(image.squeeze().numpy(), coords[j])
#             predictions.append(labels)

#         predictions = np.array(predictions)
#         ious = [get_iou(t, p) for p, t in zip(predictions, truths)]

#         scores[i] = np.mean(ious)
#         print(f'{parameter_to_test}: {parameter_value}, IoU Score: {scores[i]}')

#     print(
#         f'Max IoU: {scores.max()} with parameter_value: {parameter_range[scores.argmax()]}'
#     )

#     masker = model_class(parameter_range[scores.argmax()])
#     gt_masks = []
#     predictions = []
#     for image, gt_mask in tqdm(val_loader):
#         gt_mask = gt_mask.squeeze().numpy()
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
#             gt_mask.astype(np.uint8)
#         )
#         coordinates = [
#             (int(centroids[label][1]), int(centroids[label][0]))
#             for label in range(1, num_labels)
#         ]
#         labels = masker.predict(image.squeeze().numpy(), coordinates)
#         gt_masks.append(gt_mask)
#         predictions.append(labels)

#     gt_masks = np.array(gt_masks)
#     predictions = np.array(predictions)
#     ious = [get_iou(t, p) for p, t in zip(predictions, gt_masks)]

#     print(f'Validation IoU: {np.mean(ious)}')

#     data = [[x, y] for (x, y) in zip(parameter_range, scores)]

#     table = wandb.Table(data=data, columns=[parameter_to_test, 'F1'])
#     wandb.log(
#         {
#             f'{parameter_to_test}_f1_plot': wandb.plot.line(
#                 table, parameter_to_test, 'IoU', title=f'{parameter_to_test} vs IoU score'
#             )
#         }
#     )

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
    circle_size_range = np.arange(1, 25, 1)  # example range for circle size
    # threshold_range = np.arange(0.05, 2.001, 0.025)

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

    # # Function to compute scores for a given circle size
    def compute_scores(circle_size, data_loader):
        masker = SnakeMasker(circle_size=circle_size)
        iou_scores = []
        f1_scores = []
        for i, (image, gt_mask) in enumerate(data_loader):
            print(f'Testing image: {i}, Circle size: {circle_size}, Progress: {i/len(data_loader) * 100:.2f}%')
            image_np = image.squeeze().numpy()
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
    
    # Function to compute scores for a given circle size
    # def compute_scores(treshold, data_loader):
    #     masker = FloodMasker(tolerance=treshold)
    #     iou_scores = []
    #     f1_scores = []
    #     for i, (image, gt_mask) in enumerate(data_loader):
    #         print(f'Testing image: {i}, Threshold: {treshold}, Progress: {i/len(data_loader) * 100:.2f}%')
    #         image_np = image.squeeze().squeeze().numpy()
    #         gt_mask = gt_mask.squeeze().numpy()
    #         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    #             gt_mask.astype(np.uint8)
    #         )
    #         coords = [
    #             (int(centroids[label][0]), int(centroids[label][1]))
    #             for label in range(1, num_labels)
    #         ][0:1] # only first centroid
    #         pred_mask = masker.predict(image_np, coords).astype(bool)
    #         iou_scores.append(get_iou(gt_mask, pred_mask))
    #         f1_scores.append(get_f1_score(gt_mask, pred_mask))
    #     return np.mean(iou_scores), np.mean(f1_scores)


    # Perform parameter search for best circle size
    best_circle_size = None
    best_mean_iou = 0
    best_mean_f1 = 0
    results_df = pd.DataFrame(columns=['circle_size', 'mean_iou', 'mean_f1'])
    for circle_size in circle_size_range:
        print(f'Computing scores for circle size: {circle_size}')
        mean_iou, mean_f1 = compute_scores(circle_size, train_loader)
        wandb.log(
            {'circle_size': circle_size, 'mean_iou': mean_iou, 'mean_f1': mean_f1}
        )
        if mean_iou > best_mean_iou:
            best_circle_size = circle_size
            best_mean_iou = mean_iou
            best_mean_f1 = mean_f1
        results_df = results_df.append({'circle_size': circle_size, 'mean_iou': mean_iou, 'mean_f1': mean_f1}, ignore_index=True)
    
    results_df.to_csv('classical_search_active_contours_lidc.csv')

    print(
        f'Best circle size: {best_circle_size}, Mean IoU: {best_mean_iou:.4f}, Mean F1: {best_mean_f1:.4f}'
    )

    # Evaluate on validation set with the best circle size
    val_mean_iou, val_mean_f1 = compute_scores(best_circle_size, val_loader)
    print(
        f'Validation Mean IoU: {val_mean_iou:.4f}, Validation Mean F1: {val_mean_f1:.4f}'
    )
    
    test_mean_iou, test_mean_f1 = compute_scores(best_circle_size, test_loader)
    print(
        f'Test Mean IoU: {test_mean_iou:.4f}, Test Mean F1: {test_mean_f1:.4f}'
    )

    # Log results to WandB
    wandb.log(
        {
            'best_circle_size': best_circle_size,
            'validation_mean_iou': val_mean_iou,
            'validation_mean_f1': val_mean_f1,
        }
    )
    wandb.finish()
    
    # Perform parameter search for best circle size
    
    # best_treshold = None
    # best_mean_iou = 0
    # best_mean_f1 = 0
    # results_df = pd.DataFrame(columns=['treshold', 'mean_iou', 'mean_f1'])
    # for treshold in threshold_range:
    #     print(f'Computing scores for treshold: {treshold}')
    #     mean_iou, mean_f1 = compute_scores(treshold, train_loader)
    #     wandb.log(
    #         {'treshold': treshold, 'mean_iou': mean_iou, 'mean_f1': mean_f1}
    #     )
    #     if mean_iou > best_mean_iou:
    #         best_treshold = treshold
    #         best_mean_iou = mean_iou
    #         best_mean_f1 = mean_f1
    #     results_df = results_df.append({'treshold': treshold, 'mean_iou': mean_iou, 'mean_f1': mean_f1}, ignore_index=True)
    
    # results_df.to_csv('classical_search_flood_lidc.csv')

    # print(
    #     f'Best treshold size: {best_treshold}, Mean IoU: {best_mean_iou:.4f}, Mean F1: {best_mean_f1:.4f}'
    # )

    # # Evaluate on validation set with the best circle size
    # val_mean_iou, val_mean_f1 = compute_scores(best_treshold, val_loader)
    # print(
    #     f'Validation Mean IoU: {val_mean_iou:.4f}, Validation Mean F1: {val_mean_f1:.4f}'
    # )
    
    # test_mean_iou, test_mean_f1 = compute_scores(best_treshold, test_loader)
    # print(
    #     f'Test Mean IoU: {test_mean_iou:.4f}, Test Mean F1: {test_mean_f1:.4f}'
    # )

    # # Log results to WandB
    # wandb.log(
    #     {
    #         'best_threshold': best_treshold,
    #         'validation_mean_iou': val_mean_iou,
    #         'validation_mean_f1': val_mean_f1,
    #     }
    # )
    # wandb.finish()



if __name__ == "__main__":
    run()
