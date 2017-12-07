import metadata as meta


class Dataset:

    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test


class SplitData:

    def __init__(self, track_ids, y):
        self.track_ids = track_ids
        self.y = y


def get_data():
    meta_train_x, train_y, meta_valid_x, valid_y, meta_test_x, test_y = meta.get_metadata()
