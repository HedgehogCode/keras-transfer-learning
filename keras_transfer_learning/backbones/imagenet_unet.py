from keras import backend as K
from keras import layers

from segmentation_models.unet.builder import build_unet
from segmentation_models.backbones import get_backbone, get_feature_layers

from .layers import PadToMultiple, CropLike


def imagenet_unet(backbone_name='resnet50', weights='imagenet', padding_fix=True):

    def build(x):
        tensor = x
        if padding_fix:
            tensor = PadToMultiple(32)(tensor)
        tensor = layers.Lambda(lambda v: K.concatenate([v, v, v], axis=-1))(tensor)
        encoder_backbone = get_backbone(backbone_name,
                                        input_shape=(None, None, 3),
                                        input_tensor=tensor,
                                        weights=weights,
                                        include_top=False)
        feature_layer_names = get_feature_layers(backbone_name, n=4)
        backbone_model = build_unet(encoder_backbone, 1, feature_layer_names)
        tensor = backbone_model.get_layer('decoder_stage4_relu2').output
        if padding_fix:
            return CropLike()([tensor, x])
        return tensor

    return build
