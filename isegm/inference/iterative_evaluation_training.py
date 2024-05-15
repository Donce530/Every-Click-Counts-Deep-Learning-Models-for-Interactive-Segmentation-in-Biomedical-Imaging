import numpy as np
import torch
from isegm.inference import utils
from isegm.inference.clicker import DynamicClicker
import cv2
from tqdm import tqdm


def evaluate_dataset(dataset, predictor, logger, clicker_config, max_clicks = 20, **kwargs):

    all_ious = []

    logger.info('Running iterative NoC NoF evaluation')
    logger.info(f'Max clicks: {max_clicks}')

    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        
        clicker = DynamicClicker(sample.gt_mask,
                                 mode=clicker_config['mode'],
                                 size_range_modifier=clicker_config['size_range_modifier'])
        try:
            ious = evaluate_sample(
                sample.image,
                sample.gt_mask,
                np.zeros_like(sample.gt_mask),
                predictor,
                clicker,
                **kwargs,
            )
            all_ious.append(np.array(ious))

        except:
            logger.error(f'Error evaluating sample {index}')
            raise

    all_ious = np.array(all_ious)
    all_ious = np.nan_to_num(all_ious, nan=1)
    mean_ious = [np.mean(all_ious[:, i]) for i in range(max_clicks)]
    noc, nof = utils.compute_noc_metric(all_ious, [0.85, 0.9, 0.95], max_clicks)
    iou_error = utils.compute_scaled_iou_error(all_ious)

    return mean_ious, noc, nof, iou_error, all_ious


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
    clicker,
    max_iou_thr=1,
    pred_thr=0.49,
    min_clicks=1,
    max_clicks=20,
    start_progressive_merge_on_click=-1,
    **kwargs
):
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