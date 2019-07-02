"""Utilities for computing the Average Precision (AP) and mean Average Precision (mAP).

TODO improve naming of functions
TODO finish implementation
TODO document every function
TODO publish in own package
"""
from operator import itemgetter

import numpy as np
from scipy.ndimage import find_objects


# =================================================================================================
#      mAP interpolated
# =================================================================================================

def ap_segm_interpolated(preds, labelings, scores=None, iou_thresholds=None, compute_mean=False):
    """Computes the interpolated mean Average Precision for the given predictions.
    TODO describe formular
    TODO describe args
    """
    if iou_thresholds is None:
        iou_thresholds = [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95]
    pred_matchings, gt_segments = _match_preds_to_labelings(
        preds, labelings, scores)
    aps = [ap_matched_interpolated(pred_matchings, gt_segments, th)[0] for th in iou_thresholds]

    if compute_mean:
        return aps + [np.mean(aps)]
    return aps


def ap_matched_interpolated(preds, gt_segments, iou_threshold):
    """Computes the Average Precision for the given predictions.
    TODO describe interpolated
    TODO describe the formular

    The preds argument is supposed to be a list of tuples with the following structure:
    [
        (
            0.9,           # A score for the prediction
            'img01_seg09', # The best matching ground truth segment
            0.6            # The IoU with the best matching gt segment
        ),
        [...]
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


# =================================================================================================
#      mAP data science bowl 2018
# =================================================================================================

def ap_dsb2018(preds, labelings, iou_thresholds=None, compute_mean=False):
    """Computes the mean Average Precision as defined in the data science bowl 2018 challenge.
    TODO describe the formular

    Arguments:
        pred {list of labelings} -- A list of predictions
        labelings {list of labelings} -- A list of ground truth labelings
        iou_thresholds {list of floats} -- A list of intersection over union thresholds. The
        Average Precision will be returned for each threshold.

    Returns:
        {list} -- The Average Precision for each given threshold.
    """
    if iou_thresholds is None:
        iou_thresholds = [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95]

    aps = []
    for pred, labeling in zip(preds, labelings):
        pred_matchings, gt_segments = _match_preds_to_labelings([pred], [labeling])
        aps.append([ap_dsb2018_matched(pred_matchings, gt_segments, th)
                    for th in iou_thresholds])
    ap_means = [np.mean(ap) for ap in zip(*aps)]

    if compute_mean:
        return ap_means + [np.mean(ap_means)]
    return ap_means


def ap_dsb2018_matched(preds, gt_segments, iou_threshold):
    """Computes the Average Precision according to the data science bowl 2018 evaluation metric
    for the given predictions.

    The preds argument is supposed to be a list of tuples with the following structure:
    [
        (
            0.9,           # A score for the prediction
            'img01_seg09', # The best matching ground truth segment
            0.6            # The IoU with the best matching gt segment
        ),
        [...]
    ].

    The gt_semgents argument is a set of identifier for the ground truth segments.

    Arguments:
        preds {list of dicts} -- A list of predictions matched with the ground truth segments
                                 (See above)
        gt_segments {set} -- A set of identifiers for the ground truth segments
        iou_threshold {float} -- The IoU threshold
    """
    pred_gt = [(p[1] if p[2] > iou_threshold else None) for p in preds]
    true_positives = len({p for p in pred_gt if p in gt_segments})  # TP
    false_positives = len(preds) - true_positives  # FP
    false_negatives = len([0 for gt in gt_segments if gt not in pred_gt])  # FN
    return true_positives / (true_positives + false_positives + false_negatives)


def _match_preds_to_labelings(preds, labelings, scores=None):
    """Matches prediction segments to ground truth segments. Returns a tuble with two items:

    The first item is a list with each prediction matched to an ground truth segment:
    [
        (
            0.9,           # A score for the prediction
            'img01_seg09', # The best matching ground truth segment
            0.6            # The IoU with the best matching gt segment
        )
        [...]
    ].

    The second item is a set of identifiers for all ground truth segments.

    Arguments:
        preds {list of labelings} -- A list of predictions.
        gt_segments {list of labelings} -- A list of ground truth labelings
    """
    if scores is None:
        scores = [None] * len(preds)

    pred_matchings = []
    gt_segments = set()

    def gt_id(img_id, label_id):
        return 'img{}_lab{}'.format(img_id, label_id)

    for i, (pred, labeling, pred_scores) in enumerate(zip(preds, labelings, scores)):
        pred_objects = find_objects(pred)
        gt_objects = find_objects(labeling)

        # Loop over predicted objects
        for pred_label, pred_slice in enumerate(pred_objects, 1):
            # If there is no prediction with this id
            if pred_slice is None:
                continue

            if pred_scores is None:
                score = 1.
            else:
                score = pred_scores[pred_label]
            best_iou = 0
            best_gt = None

            # Loop over ground truth objects
            for gt_label, gt_slice in enumerate(gt_objects, 1):
                # Check if they intersect and get the union if they do
                union = _union(pred_slice, gt_slice, pred.shape)
                if union is None:
                    continue

                # Compute the IoU and check if it is the best match
                iou = _iou(pred[union], labeling[union], pred_label, gt_label)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_id(i, gt_label)

            pred_matchings.append((score, best_gt, best_iou))

        # Loop over gt objects
        for gt_label, gt_slice in enumerate(gt_objects, 1):
            gt_segments.add(gt_id(i, gt_label))

    return pred_matchings, gt_segments


def _union_1d(first, second, size):
    """Computes if slice first and second overlap for the size of the dimension"""
    first_min, first_max, _ = first.indices(size)
    second_min, second_max, _ = second.indices(size)
    if first_max >= second_min and second_max >= first_min:
        return slice(min(first_min, second_min), max(first_max, second_max))
    else:
        return None


def _union(firsts, seconds, shape):
    """Computes if slices firsts and seconds all overlap for the given shape"""
    union = []
    for first, second, size in zip(firsts, seconds, shape):
        union_1d = _union_1d(first, second, size)
        if union_1d is None:
            return None
        union.append(union_1d)
    return tuple(union)


def _iou(first, second, first_label, second_label):
    first_segm = first == first_label
    second_segm = second == second_label

    intersection = np.logical_and(first_segm, second_segm)
    union = np.logical_or(first_segm, second_segm)

    return np.sum(intersection) / np.sum(union)
