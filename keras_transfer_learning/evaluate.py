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

    results = {}
    for evalu in conf['evaluation']:
        ev_name = evalu['name']
        if ev_name == 'ap_segm_interpolated':
            res = mean_ap.ap_segm_interpolated(pred_segm, test_y, scores, **evalu['arguments'])
        elif ev_name == 'ap_dsb2018':
            res = mean_ap.ap_dsb2018(pred_segm, test_y, **evalu['arguments'])
        else:
            raise ValueError('Unknown evaluation "{}".'.format(ev_name))

        for n, r in zip(evalu['names'], res):
            results[ev_name + '#' + n] = r

    return results
