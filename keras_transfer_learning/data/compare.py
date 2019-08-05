import math

import tqdm
import numpy as np
from skimage import measure, transform
from scipy import linalg, stats
from sklearn import mixture

from keras import models

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


def model_distance(model_config: dict, data_config: dict, feature_layer_name: str,
                   distance_metrics: list, num_samples=20) -> list:

    # Compute the features for the original data and the new data
    orig_features, other_features = _model_features(
        model_config, [model_config, {'data': data_config}], feature_layer_name, num_samples)

    metric_funs = {
        'frechet': frechet_distance,
        'gmm_kl': gmm_kl,
        'gmm_js': gmm_js,
        'gmm': gmm,
    }

    res = []
    for metric in distance_metrics:
        if callable(metric):
            res.append(metric(orig_features, other_features))
        elif metric in metric_funs:
            res.append(metric_funs[metric](orig_features, other_features))
        else:
            raise ValueError(f'There is no metric for {metric}.')
    return res


def frechet_distance(features_1, features_2):
    # Compute the mean and covariance
    mean_1 = np.mean(features_1, axis=0)
    cov_1 = np.cov(features_1, rowvar=0)

    mean_2 = np.mean(features_2, axis=0)
    cov_2 = np.cov(features_2, rowvar=0)

    # Compute the frechet distance
    mean_dist = np.sum(np.square(mean_1 - mean_2))
    cov_dist = np.trace(cov_1 + cov_2 - 2 *
                        linalg.fractional_matrix_power(np.dot(cov_1, cov_2), 1/2))

    return np.real(mean_dist + cov_dist)


def gmm_kl(features_1, features_2):
    n_features = 10000
    # Only sample a few features
    state = np.random.RandomState(10)
    features_1_sample = features_1[
        state.permutation(features_1.shape[0])[:n_features], ...]
    features_2_sample = features_2[
        state.permutation(features_2.shape[0])[:n_features], ...]

    # Fit a GMM for each feature set
    gmm_1 = mixture.GaussianMixture(5)
    gmm_1.fit(features_1_sample)

    gmm_2 = mixture.GaussianMixture(5)
    gmm_2.fit(features_2_sample)

    # https://stackoverflow.com/a/26079963
    n_samples = 10**5
    samples, _ = gmm_1.sample(n_samples)
    log_1_samples = gmm_1.score_samples(samples)
    log_2_samples = gmm_2.score_samples(samples)

    return np.mean(log_1_samples) - np.mean(log_2_samples)


def gmm(features_1, features_2, n_features_gmm=10**4, n_features_scoring=10**5):
    # Only sample a few features
    state = np.random.RandomState(10)
    features_1_sample = features_1[
        state.permutation(features_1.shape[0])[:n_features_gmm], ...]

    # Fit a GMM for each feature set
    gmm_model = mixture.GaussianMixture(5)
    gmm_model.fit(features_1_sample)

    # Score the second features with the gmm
    features_2_sample = features_2[
        state.permutation(features_2.shape[0])[:n_features_scoring], ...]
    scores = gmm_model.score_samples(features_2_sample)
    scores = _remove_outliers(scores)
    scores = _remove_outliers(scores)
    scores = _remove_outliers(scores)
    return np.mean(scores)


def _remove_outliers(values, std_threshold=3):
    z_scores = np.abs(stats.zscore(values))
    return values[z_scores <= std_threshold]


def gmm_js(features_1, features_2):
    n_features = 10000
    # Only sample a few features
    features_1_sample = features_1[
        np.random.permutation(features_1.shape[0])[:n_features], ...]
    features_2_sample = features_2[
        np.random.permutation(features_2.shape[0])[:n_features], ...]

    # Fit a GMM for each feature set
    gmm_a = mixture.GaussianMixture(5)
    gmm_a.fit(features_1_sample)

    gmm_b = mixture.GaussianMixture(5)
    gmm_b.fit(features_2_sample)

    # https://stackoverflow.com/a/26079963
    n_samples = 10**5
    samples_1, _ = gmm_a.sample(n_samples)
    log_a_samples_1 = gmm_a.score(samples_1)
    log_b_samples_1 = gmm_b.score(samples_1)
    log_mix_1 = np.logaddexp(log_a_samples_1, log_b_samples_1)

    samples_2, _ = gmm_b.sample(n_samples)
    log_a_samples_2 = gmm_b.score(samples_2)
    log_b_samples_2 = gmm_b.score(samples_2)
    log_mix_2 = np.logaddexp(log_a_samples_2, log_b_samples_2)

    print('np.mean(log_a_samples_1)', log_a_samples_1)
    print('np.mean(log_b_samples_1)', log_b_samples_1)
    print('np.mean(log_mix_1)', np.mean(log_mix_1))
    print('np.mean(log_b_samples_2)', log_b_samples_2)
    print('np.mean(log_mix_2)', log_mix_2)

    return (np.mean(log_a_samples_1) - (np.mean(log_mix_1) - np.log(2))
            + np.mean(log_b_samples_2) - (np.mean(log_mix_2) - np.log(2))) / 2


def frenchet_model_distance(model_config, data_config, feature_layer_name, num_samples=20):
    return model_distance(model_config, data_config, feature_layer_name, ['frechet'], num_samples)[0]


def _model_features(model_config: dict, data_configs: list, feature_layer_name: str,
                    num_samples=20) -> list:
    # Create the feature model
    full_model = model.Model(model_config, load_weights='last')
    feature_model = utils.utils.model_up_to_layer(
        full_model.model, feature_layer_name)

    # Compute the features for each data config
    return [_model_features_for_data(feature_model, d, num_samples) for d in data_configs]


def _model_features_for_data(feature_model: models.Model, data_config: dict,
                             num_samples) -> np.ndarray:
     # Get the size of the features
    features_size = feature_model.output.shape[-1].value

    # Load the data
    data, _ = dataset.Dataset(data_config).create_test_dataset()

    # Only use some samples
    if isinstance(num_samples, int):
        data = [data[i]
                for i in np.random.permutation(len(data))[:num_samples]]
    elif isinstance(num_samples, list):
        data = [data[i] for i in num_samples]

    # Compute the features
    features_list = []
    for img in tqdm.tqdm(data):
        pred = feature_model.predict(img[None, ..., None])
        features_list.append(np.reshape(pred, (-1, features_size)))

    return np.concatenate(features_list)
