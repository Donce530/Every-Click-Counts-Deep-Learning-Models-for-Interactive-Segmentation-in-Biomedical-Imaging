from time import time

import numpy as np
import torch
import os
from isegm.inference import utils
from isegm.inference.clicker import Clicker
import shutil
import cv2
from isegm.utils.vis import add_tag
import pandas as pd


try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, logger, **kwargs):
    num_clicks = kwargs.get('max_clicks', 20)
    use_init_mask = kwargs.get('use_init_mask', False)

    all_results = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        init_mask = np.zeros_like(sample.gt_mask)
        if use_init_mask:
            init_mask = sample.init_mask

        try:
            ious = evaluate_sample(
                sample.image,
                sample.gt_mask,
                init_mask,
                predictor,
                **kwargs,
            )
        except:
            logger.error(f'Error evaluating sample {index}')
            raise

        sample_results = [index]
        sample_results.extend(ious)

        all_results.append(sample_results)

    end_time = time()
    elapsed_time = end_time - start_time

    evaluation_cols = ['sample_id']
    evaluation_cols.extend([f'iou_{i+1}_clicks' for i in range(num_clicks)])

    evaluation_results = pd.DataFrame(data=all_results, columns=evaluation_cols)
    evaluation_results.set_index('sample_id', inplace=True)

    return evaluation_results, elapsed_time


def Progressive_Merge(pred_mask, previous_mask, y, x):
    diff_regions = np.logical_xor(previous_mask, pred_mask)
    num, labels = cv2.connectedComponents(diff_regions.astype(np.uint8))
    label = labels[y, x]
    corr_mask = labels == label
    if previous_mask[y, x] == 1:
        progressive_mask = np.logical_and(previous_mask, np.logical_not(corr_mask))
    else:
        progressive_mask = np.logical_or(previous_mask, corr_mask)
    return progressive_mask


def evaluate_sample(
    image,
    gt_mask,
    init_mask,
    predictor,
    max_iou_thr,
    pred_thr=0.49,
    min_clicks=1,
    max_clicks=20,
    start_progressive_merge_on_click=1,
    **kwargs,
):
    # input_shape = kwargs.get('input_shape', (512, 512))
    # mask_shape = gt_mask.
    # original_gt_mask = gt_mask.copy()
    
    # if mask.shape != input_shape:
        
    
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    prev_mask = pred_mask
    ious_list = np.repeat(np.nan, max_clicks)
    progressive_mode = start_progressive_merge_on_click > 0    

    with torch.no_grad():
        predictor.set_input_image(image)

        predictor.set_prev_mask(init_mask)
        pred_mask = init_mask

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if progressive_mode:
                clicks = clicker.get_clicks()
                if len(clicks) >= start_progressive_merge_on_click:
                    last_click = clicks[-1]
                    last_y, last_x = last_click.coords[0], last_click.coords[1]
                    pred_mask = Progressive_Merge(pred_mask, prev_mask, last_y, last_x)
                    predictor.transforms[0]._prev_probs = np.expand_dims(
                        np.expand_dims(pred_mask, 0), 0
                    )

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list[click_indx] = iou

            prev_mask = pred_mask

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

    return ious_list
