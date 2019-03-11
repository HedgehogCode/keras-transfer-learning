"""Utilities for computing the Average Precision (AP) and mean Average Precision (mAP).

TODO improve naming of functions
TODO finish implementation
TODO document every function
TODO publish in own package
"""
from operator import itemgetter

import numpy as np


def _overlaps_1d(first, second, size):
    """Computes if slice first and second overlap for the size of the dimension"""
    first_min, first_max, _ = first.indices(size)
    second_min, second_max, _ = second.indices(size)
    return first_max >= second_min and second_max >= first_min


def _overlaps(firsts, seconds, shape):
    """Computes if slices firsts and seconds all overlap for the given shape"""
    for first, second, size in zip(firsts, seconds, shape):
        if not _overlaps_1d(first, second, size):
            return False
    return True

# def ap_segm(preds, labelings, iou_thresholds):
#     for i, (pred, labeling) in enumerate(zip(preds, labelings)):


def ap_matched_ious(preds, gt_segments, iou_thresholds):
    ap = []
    for iou_th in iou_thresholds:
        ap.append(ap_matched(preds, gt_segments, iou_th))
    return np.mean(ap), ap


def ap_matched(preds, gt_segments, iou_threshold):
    """Computes the Average Precision for the given predictions.

    The preds argument is supposed to be a list of tuples with the following structure:
    [
        (
            0.9,           # A score for the prediction
            'img01_seg09', # The best matching ground truth segment
            0.6            # The IoU with the best matching gt segment
        )
    ].

    The gt_semgents argument is a set of identifier for the ground truth segments.

    Arguments:
        preds {list of dicts} -- A list of predictions matched with the ground truth segments
                                 (See above)
        gt_segments {set} -- A set of identifiers for the ground truth segments
        iou_threshold {float} -- The IoU threshold
    """
    recall_ths = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]

    # Sort the predictions by score
    sorted_preds = sorted(preds, key=itemgetter(0), reverse=True)

    # Ground truth segments that have been predicted
    gt_unmatched = gt_segments.copy()

    matchings = []
    precision_per_recall = {}
    for recall_th in recall_ths:
        precision_per_recall[recall_th] = 0.

    current_tp = 0
    current_fp = 0

    for _, gt_id, iou in sorted_preds:
        # Check if this is a true positive
        if iou >= iou_threshold and gt_id in gt_unmatched:
            current_tp += 1
            gt_unmatched.remove(gt_id)
        else:
            current_fp += 1

        # Compute current TP, FP, FN, Precision and Recall
        true_positive = current_tp
        false_positive = current_fp
        false_negative = len(gt_unmatched)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        matchings.append(
            {
                'score': _,
                'TP': true_positive,
                'FP': false_positive,
                'FN': false_negative,
                'Precision': precision,
                'Recall': recall
            }
        )

        # Update the precision pre recall
        smaller_recalls = (r for r in recall_ths if r <= recall)
        for recall_th in smaller_recalls:
            precision_per_recall[recall_th] = max(
                precision_per_recall[recall_th], precision)

    average_precision = np.mean(list(precision_per_recall.values()))
    return average_precision, matchings
