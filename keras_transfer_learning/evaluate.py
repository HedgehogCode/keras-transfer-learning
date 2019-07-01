import numpy as np

from keras_transfer_learning import model, dataset
from mean_average_precision import mean_ap


def evaluate(conf, epoch=None):
    # Create the mdoel
    if epoch is None:
        m = model.Model(conf, load_weights='last')
    else:
        m = model.Model(conf, load_weights='epoch', epoch=epoch)

    # TODO implement
    if conf['data']['name'] == 'cityscapes':
        raise ValueError(
            'Evaluation is not supported for the cityscapes dataset yet.')

    # Create the dataset
    d = dataset.Dataset(conf)
    test_x, test_y = d.create_test_dataset()

    # Run the prediction
    pred = m.predict_and_process(test_x)
    pred_segm, scores = zip(*pred)  # unzip

    # TODO allow for different evaluations
    # Evaluate
    iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70,
                      0.75, 0.80, 0.85, 0.90, 0.95]
    ap = mean_ap.ap_segm_interpolated(
        pred_segm, test_y, scores, iou_thresholds=iou_thresholds)
    print("The average precision is {}".format(np.mean(ap)))

    return {k: v for k, v in zip(iou_thresholds, ap)}
