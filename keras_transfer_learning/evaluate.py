from keras_transfer_learning import model, dataset
from keras_transfer_learning.utils import mean_average_precision


def evaluate(conf):
    # TODO allow loading a specific epoch
    # Create the mdoel
    m = model.Model(conf, load_weights='last')

    # Create the dataset
    d = dataset.Dataset(conf)
    test_x, test_y = d.create_test_dataset()

    # Run the prediction
    pred = m.predict_and_process(test_x)

    # TODO allow for different evaluations
    # Evaluate
    iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    ap = mean_average_precision.ap_segm(pred, test_y, iou_thresholds)

    print("The average precision is {}".format(ap))
    return ap
