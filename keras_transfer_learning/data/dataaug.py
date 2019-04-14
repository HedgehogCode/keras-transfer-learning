from imgaug import augmenters as iaa


def create_imgaug_augmentor(augmentors):
    augs = []
    for aug in augmentors:
        augs.append(getattr(iaa, aug['name'])(**aug['args']))
    augmentor = iaa.Sequential(augs)

    def augment(data):
        aug_det = augmentor.to_deterministic()
        augmented = []
        for batch in data:
            augmented.append(aug_det.augment_images(batch))
        return augmented
