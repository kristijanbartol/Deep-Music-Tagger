import numpy as np
from PIL import Image

import math
import sys
import os
import random

import metadata as meta

spectr_template = '../../in/mel-specs/{}'

img_height = 96
img_width  = 1366

idx_map = dict()


class MultiClassDataset:
    """
    First layer of dataset class. The idea is to
    "encapsulate" logic inside each of data splits.

    The output vector is one-hot encoded, i.e.,
    only one class is true for each sample.
    """

    number_of_classes = 16
    genre_to_id_map = dict()
    id_to_genre_map = dict()
    genres_map = { 3: 'Blues', 5: 'Classical', 9: 'Country', 13: 'Easy Listening',
            15: 'Electronic', 38: 'Experimental', 17: 'Folk', 21: 'Hip-Hop',
            1234: 'Instrumental', 2: 'International', 4: 'Jazz', 8: 'Old-Time / Historic',
            10: 'Pop', 12: 'Rock', 14: 'Soul-RnB', 20: 'Spoken' }

    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, test_y):

        def _create_idx_map(train_top, valid_top, test_top):
            current_id = 0
            for genre_id in np.hstack((train_top, valid_top, test_top)):
                if genre_id not in self.genre_to_id_map:
                    # genre_id should also be the key in genres_map
                    self.genre_to_id_map[genre_id] = current_id 
                    self.id_to_genre_map[current_id] = genre_id
                    current_id += 1

        _create_idx_map(train_y[0], valid_y[0], test_y[0])
        self.train = SplitData(train_x, train_y[0], 'train')
        self.valid = SplitData(valid_x, valid_y[0], 'valid')
        self.test = SplitData(test_x, test_y[0], 'test')


class SplitData:
    """
    Inner layer that does the actual spectrogram images fetching
    for each batch and assigns expected values to output vector y.
    """

    def __init__(self, track_ids, y_top, dataset_label):
        self.current_sample_idx = 0
        self.track_ids = track_ids
        self.labels = self._create_output_vector(y_top)
        self.dataset_label = dataset_label

    def _create_output_vector(self, y_top):
        """
        The output vector is one-hot encoded according
        to provided :param y_top:

        :param y_top:
        :return y:
        """

        def _transform_idx(y):
            return [MultiClassDataset.genre_to_id_map[yi] for yi in y]

        return np.eye(MultiClassDataset.number_of_classes)[_transform_idx(y_top)]

    def _load_images(self, track_ids):
        """
        Private method for actual loading spectrogram data.

        :param track_ids:
        :return images:
        """
        images = []
        for track_id in track_ids:
            fpath = spectr_template.format(track_id[:3] + '/' + track_id + '.png')
            print('Loading spectrogram: {} ({})'.format(fpath, self.dataset_label))
            images.append(np.asarray(Image.open(fpath).getdata()).reshape(img_width, img_height))
        return np.array(images)

    def load_all(self):
        """
        Returns all the data with loaded spectrograms.

        :return images, labels:
        """
        return self._load_images(self.track_ids), self.labels

    def get_number_of_batches(self, batch_size):
        """
        :return number_of_batches:
        """
        return int(math.ceil(self.track_ids.shape[0] / batch_size))

    def next_batch(self, batch_size):
        """
        Takes subset of input and output for interval
        (current_idx : current_idx + batch_size).

        :param batch_size:
        :return batch_images, batch_labels:
        """
        if self.current_sample_idx + batch_size >= self.track_ids.shape[0]:  # edge case when latter index is overflown
            filling_ids = random.sample(range(self.current_sample_idx),
                                        batch_size - (self.track_ids.shape[0] - self.current_sample_idx))
            batch_images = self._load_images(
                self.track_ids[list(range(self.current_sample_idx, self.track_ids.shape[0])) + filling_ids])
            batch_labels = self.labels[list(range(self.current_sample_idx, self.labels.shape[0])) + filling_ids]
            self.current_sample_idx = 0
        else:
            batch_images = self._load_images(
                self.track_ids[self.current_sample_idx:self.current_sample_idx + batch_size])
            batch_labels = self.labels[self.current_sample_idx:self.current_sample_idx + batch_size]
            self.current_sample_idx += batch_size

        return batch_images, batch_labels

    def shuffle(self):
        indices = np.arange(self.track_ids.shape[0])
        np.random.shuffle(indices)

        self.track_ids = self.track_ids[indices]
        self.labels = self.labels[indices]


def get_data():
    """
    Reads metadata, stacks input and output to a single object.
    Returns complex object that contain splitted set objects.

    X vectors contain track ids, not spectrograms, that is why
    they are prefixed with 'meta_' prefix.
    Y vectors are structured as ([[top_genre], [all_genres]]).

    :return Dataset(train, test, valid):
    """

    def _clean_track_ids(track_ids, labels):
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
                labels = np.delete(labels, idx - dlt_cnt, 1)
                dlt_cnt += 1

        return track_ids, labels, all_cnt, dlt_cnt

    meta_train_x, train_y, meta_valid_x, valid_y, meta_test_x, test_y = meta.get_metadata()

    meta_train_x, train_y, all_cnt, dlt_cnt = _clean_track_ids(meta_train_x, train_y)
    print('Removed {} of {} train records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))
    meta_test_x, test_y, all_cnt, dlt_cnt = _clean_track_ids(meta_test_x, test_y)
    print('Removed {} of {} test records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))
    meta_valid_x, valid_y, all_cnt, dlt_cnt = _clean_track_ids(meta_valid_x, valid_y)
    print('Removed {} of {} validation records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))

    return MultiClassDataset(meta_train_x, train_y, meta_test_x, test_y, meta_valid_x, valid_y)


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

    data.test.load_all()
