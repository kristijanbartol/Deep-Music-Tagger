import numpy as np
import os
import random
from PIL import Image

import metadata as meta

spectr_template = '../in/mel-specs/{}'


class Dataset:
    """
    First dataset class layer. The idea is to
    "encapsulate" logic inside each of data splits.
    """

    def __init__(self, train, valid, test):
        self.train = SplitData(train[0], train[1], train[2])
        self.valid = SplitData(valid[0], valid[1], valid[2])
        self.test = SplitData(test[0], test[1], test[2])


class SplitData:
    """
    Inner layer that does the actual spectrogram images fetching
    for each batch and assigns expected values to output vector y.
    """

    def __init__(self, track_ids, y_top, y_all):
        self.top_genre_significance = 0.75
        self.current_sample_idx = 0
        self.track_ids = track_ids
        self.y = self._create_output_vector(y_top, y_all)

    @staticmethod
    def _get_indices_mapping(y_all):
        indices = []
        for genre_list in y_all:
            for genre_id in genre_list:
                if genre_id not in indices:
                    indices.append(genre_id)
        indices = sorted(indices)
        indices_map = dict()
        for i, index in enumerate(indices):
            indices_map[index] = i
        return indices_map

    def _create_output_vector(self, y_top, y_all):
        idx_map = self._get_indices_mapping(y_all)

        # dim(y) = (number_of_samples, number_of_unique_indices)
        y = []
        vsize = len(idx_map)
        for i in range(y_top.shape[0]):
            yi = [0] * vsize
            yi[idx_map[y_top[i]]] = self.top_genre_significance if len(y_all[i]) > 1 else 1

            other_genres_significance = (1 - self.top_genre_significance) / max(1, len(y_all[i]) - 1)
            for genre_id in y_all[i]:
                if genre_id == y_top[i]:
                    continue
                yi[idx_map[genre_id]] = other_genres_significance
            y.append(yi)

        return np.array(y)

    @staticmethod
    def _load_images(track_ids):
        images = []
        for track_id in track_ids:
            fpath = spectr_template.format(track_id[:3] + '/' + track_id + '.png')
            images.append(np.asarray(Image.open(fpath).getdata()).reshape(1406, 96))
        return np.array(images)

    def next_batch(self, batch_size):
        # edge case when latter index is overflown
        if self.current_sample_idx + batch_size > self.track_ids.shape[0]:
            filling_ids = random.sample(range(self.current_sample_idx),
                                        self.track_ids.shape[0] - self.current_sample_idx)
            batch_track_ids = self._load_images(
                self.track_ids[list(range(self.current_sample_idx, self.track_ids.shape[0])) + filling_ids])
            self.current_sample_idx = 0
        else:
            print(self.current_sample_idx)
            print(self.track_ids.shape[0])
            batch_track_ids = self._load_images(
                self.track_ids[self.current_sample_idx:self.current_sample_idx + batch_size])
            self.current_sample_idx += batch_size
        batch_y = self.y[self.current_sample_idx:self.current_sample_idx + batch_size]

        return batch_track_ids, batch_y


def _vstack(track_ids, y_stack):
    """
    Some spectrogram images might be missing as they
    failed to generate so dimensions wouldn't match
    if regular np.hstack is used. This function removes
    rows that would contain missing spectrograms.

    :param images:
    :param y_stack:
    :return:
    """
    all_cnt = 0
    dlt_cnt = 0
    for idx, track_id in enumerate(track_ids):
        track_id_str = str(track_id)
        all_cnt += 1
        if not os.path.isfile(spectr_template.format(track_id_str[:3] + '/' + track_id_str + '.png')):
            print(spectr_template.format(track_id_str[:3] + '/' + track_id_str + '.png'))
            np.delete(track_ids, idx, 0)
            np.delete(y_stack, idx, 1)
            dlt_cnt += 1
    return np.vstack((track_ids, y_stack)), all_cnt, dlt_cnt


def get_data():
    """
    Reads metadata, stacks input and output to a single object.
    Returns complex object that contain splitted set objects.

    :return Dataset(train, test, valid):
    """
    meta_train_x, train_y, meta_valid_x, valid_y, meta_test_x, test_y = meta.get_metadata()

    train, all_cnt, dlt_cnt = _vstack(meta_train_x, train_y)
    print('Removed {} of {} train records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))
    test, all_cnt, dlt_cnt = _vstack(meta_test_x, test_y)
    print('Removed {} of {} test records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))
    valid, all_cnt, dlt_cnt = _vstack(meta_valid_x, valid_y)
    print('Removed {} of {} validation records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))

    return Dataset(train, test, valid)


if __name__ == '__main__':
    """
    Used for testing.
    """
    data = get_data()
    batch_size = 100
    for i in range(100):
        batch = data.test.next_batch(batch_size)
        print(batch[0].shape)
