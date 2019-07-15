from keras import layers

from segmentation_models.unet.builder import build_unet
from segmentation_models.backbones import get_preprocessing, get_backbone, get_feature_layers

BACKBONE = 'resnet50'

preprocess_input = get_preprocessing(BACKBONE)

inp = layers.Input((None, None, 3))

encoder_backbone = get_backbone(BACKBONE,
                                input_shape=(None, None, 3),
                                input_tensor=inp,
                                weights='imagenet',
                                include_top=False)
backbone_model = build_unet(encoder_backbone, 1, get_feature_layers(BACKBONE, n=4))
backbone_model.summary(line_length=150)

feature_layer = backbone_model.get_layer('decoder_stage4_relu2')

print('Ende gelaende')
