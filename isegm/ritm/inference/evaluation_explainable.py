from time import time

import numpy as np
import torch

from isegm.ritm.inference import utils
from isegm.inference.clicker import DynamicClicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, clicker_config, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        clicker = DynamicClicker(sample.gt_mask,
                                 mode=clicker_config['mode'],
                                 size_range_modifier=clicker_config['size_range_modifier'])
        _, sample_ious, _ = evaluate_sample(
            sample.image, sample.gt_mask, predictor, clicker, sample_id=index, **kwargs
        )
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample(
    image,
    gt_mask,
    predictor,
    clicker,
    max_iou_thr,
    pred_thr=0.5,
    min_clicks=1,
    max_clicks=20,
    sample_id=None,
    callback=None,
):
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    perdicted_masks = []
    info_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs, info = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr
            perdicted_masks.append(pred_mask)

            if callback is not None:
                callback(
                    image,
                    gt_mask,
                    pred_probs,
                    sample_id,
                    click_indx,
                    clicker.clicks_list,
                )

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)
            info_list.append(info)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return (
            clicker.clicks_list,
            np.array(ious_list, dtype=np.float32),
            pred_probs,
            perdicted_masks,
            info_list,
        )
