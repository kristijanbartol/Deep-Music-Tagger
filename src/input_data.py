import numpy as np
import os

import metadata as meta

spectr_template = '../in/mel-specs/{}'


class Dataset:

    def __init__(self, train, valid, test):
        self.train = SplitData(train[0], train[1], train[2])
        self.valid = SplitData(valid[0], valid[1], valid[2])
        self.test = SplitData(test[0], test[1], test[2])


class SplitData:

    def __init__(self, track_ids, y_oh, y_multi):
        self.track_ids = track_ids
        self.y_oh = y_oh
        self.y_multi = y_multi
        self.y = y_oh       # problem is single label classification in the first phase

    def next_batch(self, batch_size):
        pass


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
    meta_train_x, train_y, meta_valid_x, valid_y, meta_test_x, test_y = meta.get_metadata()

    train, all_cnt, dlt_cnt = _vstack(meta_train_x, train_y)
    print('Removed {} of {} train records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))
    test, all_cnt, dlt_cnt = _vstack(meta_test_x, test_y)
    print('Removed {} of {} test records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))
    valid, all_cnt, dlt_cnt = _vstack(meta_valid_x, valid_y)
    print('Removed {} of {} validation records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))

    return Dataset(train, test, valid)


if __name__ == '__main__':
    get_data()
