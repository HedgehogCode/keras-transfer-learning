from imgaug import augmenters as iaa
import imgaug as ia


def create_imgaug_augmentor(augmentors):
    augs = []
    augs_seg = ['root']
    for idx, aug in enumerate(augmentors):
        a = getattr(iaa, aug['name'])(name=str(idx), **aug['args'])
        augs.append(a)
        if aug.get('onsegm', True):
            augs_seg.append(str(idx))
    augmentor_seq = iaa.Sequential(augs, name='root')

    def segm_activator(images, augmentor, parents, default):
        return default if augmentor.name in augs_seg else False
    segm_hook = ia.HooksImages(activator=segm_activator)

    def augment(batch_x, batch_y):
        aug_det = augmentor_seq.to_deterministic()

        batch_x_aug = aug_det.augment_images(batch_x)
        batch_y_aug = aug_det.augment_images(batch_y, hooks=segm_hook)

        return batch_x_aug, batch_y_aug
    return augment
