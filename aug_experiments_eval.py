#!/usr/bin/env python

import sys
import argparse

import keras.backend as K

from keras_transfer_learning import evaluate, utils


def main(args):

    models = [
        'models/Z/02_resnet-unet_stardist_dsb2018-aug-0gamma_R_F/config.yaml',
        'models/Z/02_resnet-unet_stardist_dsb2018-aug-1sharpen_R_F/config.yaml',
        'models/Z/02_resnet-unet_stardist_dsb2018-aug-2emboss_R_F/config.yaml',
        'models/Z/02_resnet-unet_stardist_dsb2018-aug-3add_R_F/config.yaml',
        'models/Z/02_resnet-unet_stardist_dsb2018-aug-4gaussnoise_R_F/config.yaml',
        'models/Z/02_resnet-unet_stardist_dsb2018-aug-5gaussblur_R_F/config.yaml',
        'models/Z/02_resnet-unet_stardist_dsb2018-aug-6motion_R_F/config.yaml',
        'models/Z/02_resnet-unet_stardist_dsb2018-aug-7invert_R_F/config.yaml'
    ]

    results = []
    for model in models:
        K.clear_session()
        conf = utils.utils.yaml_load(model)
        res = evaluate.evaluate(conf)
        print('Result:', res)
        results.append(res)

    print(results)
    print('----------------------------------------------')
    print('----------------------------------------------')

    print([(m.split('/')[2].split('_')[3], r['ap_dsb2018#mean']) for m, r in zip(models, results)])

    # Results:
    # dsb2018-aug-3add:        0.4692157527363988
    # dsb2018-aug-1sharpen:    0.45248331193684227
    # dsb2018-aug-0gamma:      0.4278334011570243
    # dsb2018-aug-7invert:     0.4265673324803328
    # dsb2018-aug-6motion:     0.4150465814871196
    # dsb2018-aug-5gaussblur:  0.4058868451523483
    # dsb2018-aug-4gaussnoise: 0.4031449444163961
    # dsb2018-aug-2emboss:     0.35521271819485023
    return 0


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
