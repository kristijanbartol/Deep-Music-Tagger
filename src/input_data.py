import metadata as meta


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


def _load_images(meta_set_x):
    """
    Load actual spectrogram images from
    provided metadata tracks ids.

    :param meta_set_x:
    :return:
    """
    return _


def _hstack(images, y_stack):
    """
    Some spectrogram images might be missing as they
    failed to generate so dimensions wouldn't match
    if regular np.hstack is used. This function removes
    rows that would contain missing spectrograms.

    :param images:
    :param y_stack:
    :return:
    """
    return _


def get_data():
    meta_train_x, train_y, meta_valid_x, valid_y, meta_test_x, test_y = meta.get_metadata()
    train = _hstack(_load_images(meta_train_x), train_y)
    test = _hstack(_load_images(meta_test_x), test_y)
    valid = _hstack(_load_images(meta_valid_x), valid_y)

    return Dataset(train, test, valid)
