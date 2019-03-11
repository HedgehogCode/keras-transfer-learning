import pytest

from keras_transfer_learning.utils.mean_average_precision import ap_matched


def test_ap_easy():
    """Tests an easy average precicion example"""
    preds = [
        (0.5, 'gt2', 0.60),  # TRUE,  Close match
        (0.9, 'gt1', 0.85),  # TRUE,  Good match
        (0.7, 'gt4', 0.30),  # FALSE, Low iou
        (0.2, 'gt0', 0.60),  # FALSE, Already matched with higher score
        (1.0, 'gt0', 0.90),  # TRUE,  Good match
        (0.3, None, 0.00),  # FALSE, No match
        (0.4, 'gt3', 0.90),  # TRUE,  Good match
        (0.6, 'gt1', 0.80),  # FALSE, Already matched with higher score
        (0.8, None, 0.00),  # FALSE, No match
        (0.1, 'gt4', 0.80)  # TRUE,  Good match
    ]
    gt_segments = ['gt0', 'gt1', 'gt2', 'gt3', 'gt4']
    iou_th = 0.6

    ap, matchings = ap_matched(preds, gt_segments, iou_th)

    assert pytest.approx(0.7532467532467532, 0.00001) == ap
    # TODO check matchings
