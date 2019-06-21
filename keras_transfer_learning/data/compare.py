import math

import tqdm
import numpy as np
from skimage import measure, transform
from scipy import linalg

from keras_transfer_learning import utils, dataset, model


def ssim(im1, im2):
    # TODO do not crop one image?
    max_shape = []
    for x1, x2 in zip(im1.shape, im2.shape):
        max_shape.append(slice(0, min(x1, x2)))
    im1 = im1[tuple(max_shape)]
    im2 = im2[tuple(max_shape)]
    return measure.compare_ssim(im1, im2)


def ssim_datasets(A, B, samples=1000):
    scores = []
    for _ in range(samples):
        first_idx = np.random.randint(0, len(A))
        second_idx = np.random.randint(0, len(B))
        scores.append(ssim(A[first_idx], B[second_idx]))
    return np.mean(scores)

# Changed from
# https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py


# TODO remove
# def get_descriptors_for_dataset(minibatch, nhood_size, nhoods_per_image):
#     S = minibatch.shape # (minibatch, height, width)
#     assert len(S) == 3
#     N = nhoods_per_image * S[0]
#     H = nhood_size // 2
#     nhood, x, y = np.ogrid[0:N, -H:H+1, -H:H+1]
#     img = nhood // nhoods_per_image
#     x = x + np.random.randint(H, S[2] - H, size=(N, 1, 1))
#     y = y + np.random.randint(H, S[1] - H, size=(N, 1, 1))
#     idx = (img * S[1] + y) * S[2] + x
#     return minibatch.flat[idx]

def sliced_wasserstein_pyramids(A, B, **kwargs):
    pyramid_levels = kwargs.get('pyramid_levels', 4)
    if A is not B:
        pyramids_a = laplace_pyramids(A, pyramid_levels)
        pyramids_b = laplace_pyramids(B, pyramid_levels)
    else:
        pyramids_a = pyramids_b = laplace_pyramids(A, pyramid_levels)

    results = []
    for level_a, level_b in zip(pyramids_a, pyramids_b):
        results.append(sliced_wasserstein_patches(level_a, level_b, **kwargs))

    return results


def sliced_wasserstein_patches(A, B, **kwargs):
    patch_size = kwargs.get('patch_size', 7)
    n_patches = kwargs.get('n_patches', 2**21)
    dir_repeats = kwargs.get('dir_repeats', 4)
    dirs_per_repeat = kwargs.get('dirs_per_repeat', 128)

    # Get the descriptors
    desc_a = get_descriptors_for_dataset(
        A, patch_size, math.ceil(n_patches / len(A)))
    np.random.shuffle(desc_a)
    desc_a = desc_a[:n_patches]

    desc_b = get_descriptors_for_dataset(
        B, patch_size, math.ceil(n_patches / len(B)))
    np.random.shuffle(desc_b)
    desc_b = desc_b[:n_patches]

    # Compute the sliced wasserstein distance
    return sliced_wasserstein(desc_a, desc_b, dir_repeats, dirs_per_repeat)


def get_descriptors_for_dataset(dataset, patch_size, patches_per_image):
    # TODO test that patches are real
    H = patch_size // 2
    x_patch_grid = np.ogrid[-H:H+1][None, None, :]
    y_patch_grid = np.ogrid[-H:H+1][None, :, None]

    desc = np.empty((len(dataset) * patches_per_image, patch_size, patch_size))
    for i, img in enumerate(dataset):
        y = y_patch_grid + \
            np.random.randint(
                H, img.shape[0] - H, size=(patches_per_image, 1, 1))
        x = x_patch_grid + \
            np.random.randint(
                H, img.shape[1] - H, size=(patches_per_image, 1, 1))
        idx = y * img.shape[1] + x
        desc[(i*patches_per_image):((i+1)*patches_per_image)] = img.flat[idx]
    return desc.reshape((desc.shape[0], -1))


def sliced_wasserstein(A, B, dir_repeats=4, dirs_per_repeat=128):
    # (neighborhood, descriptor_component)
    assert A.ndim == 2 and A.shape == B.shape
    results = []
    for _ in range(dir_repeats):
        # (descriptor_component, direction)
        dirs = np.random.randn(A.shape[1], dirs_per_repeat)
        # normalize descriptor components for each direction
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True))
        dirs = dirs.astype(np.float32)
        # (neighborhood, direction)
        projA = np.matmul(A, dirs)
        projB = np.matmul(B, dirs)
        # sort neighborhood projections for each direction
        projA = np.sort(projA, axis=0)
        projB = np.sort(projB, axis=0)
        # pointwise wasserstein distances
        dists = np.abs(projA - projB)
        # average over neighborhoods and directions
        results.append(np.mean(dists))
    # average over repeats
    return np.mean(results)


def laplace_pyramids(dataset, levels=4):
    pyramids = [[] for _ in range(levels)]
    for img in dataset:
        pyramid = transform.pyramid_laplacian(
            img, max_layer=levels-1, multichannel=False)
        for i, img_level in enumerate(pyramid):
            pyramids[i].append(img_level)

    return pyramids


def frechet_distance(mean_1, cov_1, mean_2, cov_2):
    return np.sum(np.square(mean_1 - mean_2)) + \
        np.trace(cov_1 + cov_2 - 2 *
                 linalg.fractional_matrix_power(np.dot(cov_1, cov_2), 1/2))


def frenchet_model_distance(model_config, data_config, feature_layer_name, num_samples=20):
    # Create the feature model
    full_model = model.Model(model_config, load_weights='last')
    feature_model = utils.utils.model_up_to_layer(
        full_model.model, feature_layer_name)
    features_size = feature_model.output.shape[-1].value

    def get_mgf(data):
        if num_samples is not None:
            data = [data[i]
                    for i in np.random.permutation(len(data))[:num_samples]]

        features_list = []
        for img in tqdm.tqdm(data):
            pred = feature_model.predict(img[None, ..., None])
            features_list.append(np.reshape(pred, (-1, features_size)))

        features = np.concatenate(features_list)
        mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=0)
        return mean, cov

    # Compute a mgf on the original model data
    orig_data = dataset.Dataset(model_config)
    orig_x, _ = orig_data.create_test_dataset()
    orig_mean, orig_cov = get_mgf(orig_x)

    # Compute a mgf on the new dataset
    other_data = dataset.Dataset({'data': data_config})
    other_x, _ = other_data.create_test_dataset()
    other_mean, other_cov = get_mgf(other_x)

    # Compute the frenchet distance
    return frechet_distance(orig_mean, orig_cov, other_mean, other_cov)
