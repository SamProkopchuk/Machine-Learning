import numpy as np


def discretize(features, feat_ranges, buckets_per_feat):
    features, feat_ranges, buckets_per_feat = map(
        np.asarray, (features, feat_ranges, buckets_per_feat))
    features = (features - feat_ranges[:, 0]) / \
        (feat_ranges[:, 1] - feat_ranges[:, 0]) * buckets_per_feat
    return features.astype(int)


def tile_encode(samples, feat_ranges, buckets_per_feat):
    samples, feat_ranges, buckets_per_feat = map(
        np.asarray, (samples, feat_ranges, buckets_per_feat))
    tile_shape = len(feat_ranges) * [buckets_per_feat] if buckets_per_feat.size == 1 else buckets_per_feat
    res = np.zeros((len(samples), *tile_shape))
    dis = np.apply_along_axis(discretize, 1, samples, feat_ranges, buckets_per_feat)
    res[np.arange(len(samples))[...], dis[..., 0], dis[..., 1]] = 1
    return res


def hash_encode(samples, feat_ranges, buckets_per_feat, hash_table_size, hash_fn=hash):
    samples, feat_ranges, buckets_per_feat = map(
        np.asarray, (samples, feat_ranges, buckets_per_feat))
    tile_shape = len(feat_ranges) * [buckets_per_feat] if buckets_per_feat.size == 1 else buckets_per_feat
    res = np.zeros((len(samples), hash_table_size))
    dis = np.apply_along_axis(discretize, 1, samples, feat_ranges, buckets_per_feat)
    dis = np.ravel_multi_index([dis[..., 0], dis[..., 1]], tile_shape) % hash_table_size
    res[np.arange(len(samples))[...], dis] = 1
    return res


def main():

    print(discretize([0, 1], [[0, 2], [0, 2]], 4))

    print('-' * 10)

    print(tile_encode([[0.5, 1.4], [0, 0]], [[0, 2], [0, 2]], 4))

    print('-' * 10)

    print(hash_encode([[0.5, 1.4], [0, 0]], [[0, 2], [0, 2]], 4, 7))


if __name__ == '__main__':
    main()
