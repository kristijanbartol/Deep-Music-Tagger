import numpy as np
import os
import random
from PIL import Image

import metadata as meta

spectr_template = '../in/mel-specs/{}'

img_height = 96
img_width  = 1366


class Dataset:
    """
    First dataset class layer. The idea is to
    "encapsulate" logic inside each of data splits.
    """

    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, test_y):
        self.train = SplitData(train_x, train_y[0], train_y[1])
        self.valid = SplitData(valid_x, valid_y[0], valid_y[1])
        self.test = SplitData(test_x, test_y[0], test_y[1])


class SplitData:
    """
    Inner layer that does the actual spectrogram images fetching
    for each batch and assigns expected values to output vector y.
    """

    def __init__(self, track_ids, y_top, y_all):
        self.top_genre_significance = 0.75
        self.current_sample_idx = 0
        self.track_ids = track_ids
        self.labels = self._create_output_vector(y_top, y_all)

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
        """
        Instead of typical one-hot vector, a vector with
        more than single non-zero element is created for
        multi-output classification.

        For top_genre, top_genre_significance is assigned.
        For the rest of genres, if available,
        (1 - top_genre_significance) is evenly assigned.

        :param y_top:
        :param y_all:
        :return y:
        """

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
        """
        Private method for actual loading spectrogram data.

        :param track_ids:
        :return images:
        """
        images = []
        for track_id in track_ids:
            fpath = spectr_template.format(track_id[:3] + '/' + track_id + '.png')
            print('Loading spectrogram: {}'.format(fpath))
            images.append(np.asarray(Image.open(fpath).getdata()).reshape(img_width, img_height))
        return np.array(images)

    def all_loaded(self):
        """
        Returns all the data with loaded spectrograms.

        :return images, labels:
        """
        return self._load_images(self.track_ids), self.labels

    def get_output_size(self):
        """
        Returns label vector length, i.e. the number of classes.

        :return:
        """
        return self.labels.shape[1]

    def next_batch(self, batch_size):
        """
        Takes subset of input and output for interval
        (current_idx : current_idx + batch_size).

        :param batch_size:
        :return batch_images, batch_labels:
        """
        if self.current_sample_idx + batch_size > self.track_ids.shape[0]:  # edge case when latter index is overflown
            filling_ids = random.sample(range(self.current_sample_idx),
                                        self.track_ids.shape[0] - self.current_sample_idx)
            batch_images = self._load_images(
                self.track_ids[list(range(self.current_sample_idx, self.track_ids.shape[0])) + filling_ids])
            self.current_sample_idx = 0
        else:
            batch_images = self._load_images(
                self.track_ids[self.current_sample_idx:self.current_sample_idx + batch_size])
            self.current_sample_idx += batch_size
        batch_labels = self.labels[self.current_sample_idx:self.current_sample_idx + batch_size]

        print(batch_images.shape)
        return batch_images, batch_labels

    def shuffle(self):
        indices = np.arange(self.track_ids.shape[0])
        np.random.shuffle(indices)

        self.track_ids = self.track_ids[indices]
        self.labels = self.labels[indices]


def _clean_track_ids(track_ids):
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
            track_ids = np.delete(track_ids, idx - dlt_cnt, 0)
            dlt_cnt += 1

    return track_ids, all_cnt, dlt_cnt


def get_data():
    """
    Reads metadata, stacks input and output to a single object.
    Returns complex object that contain splitted set objects.

    :return Dataset(train, test, valid):
    """
    meta_train_x, train_y, meta_valid_x, valid_y, meta_test_x, test_y = meta.get_metadata()

    train_x, all_cnt, dlt_cnt = _clean_track_ids(meta_train_x)
    print('Removed {} of {} train records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))
    test_x, all_cnt, dlt_cnt = _clean_track_ids(meta_test_x)
    print('Removed {} of {} test records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))
    valid_x, all_cnt, dlt_cnt = _clean_track_ids(meta_valid_x)
    print('Removed {} of {} validation records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))

    return Dataset(train_x, train_y, test_x, test_y, valid_x, valid_y)


if __name__ == '__main__':
    """
    Used for testing and debugging.
    """
    data = get_data()
    batch_size = 100
    for i in range(100):
        #batch = data.test.next_batch(batch_size)
        pass

    print(data.test.track_ids)
    data.test.shuffle()
    print(data.test.track_ids)

    data.test.all_loaded()

