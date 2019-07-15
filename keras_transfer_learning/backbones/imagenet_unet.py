from keras import backend as K
from keras import layers

from segmentation_models.unet.builder import build_unet
from segmentation_models.backbones import get_backbone, get_feature_layers

from .layers import PadToMultiple, CropLike


def imagenet_unet(backbone_name='resnet50', weights='imagenet'):

    def build(x):
        x_padded = PadToMultiple(32)(x)
        x_padded = layers.Lambda(lambda v: K.concatenate([v, v, v], axis=-1))(x_padded)
        encoder_backbone = get_backbone(backbone_name,
                                        input_shape=(None, None, 3),
                                        input_tensor=x_padded,
                                        weights=weights,
                                        include_top=False)
        feature_layer_names = get_feature_layers(backbone_name, n=4)
        backbone_model = build_unet(encoder_backbone, 1, feature_layer_names)
        backbone_output = backbone_model.get_layer('decoder_stage4_relu2').output
        return CropLike()([backbone_output, x])

    return build
