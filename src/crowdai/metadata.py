import pandas as pd
import numpy as np
import ast
import sys

fdata_template = '../../data/fma_metadata/{}'
modeldata_template = '../../in/metadata/{}'
graphs_template = '../../out/graphs/{}'

genres_fpath = fdata_template.format('genres.csv')
tracks_fpath = fdata_template.format('tracks.csv')


class DataSize:
    """
    Helper class used to compare string values:
    'small', 'medium' and 'large' more conveniently.
    """

    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'

    def __init__(self, size='medium'):
        self.size = size

    def __ge__(self, s2):
        if self.size == self.LARGE:
            return True
        if self.size == self.MEDIUM:
            if s2 == self.LARGE:
                return False
            else:
                return True
        if self.size == self.SMALL:
            if s2 == self.MEDIUM or s2 == self.LARGE:
                return False
            else:
                return True


data_size = DataSize('medium')


def _read_metaset(fpath):
    """
    Reading metaset columns as x and y representing input and ouput.

    :param fpath:
    :return metaset_x, metaset_y:
    """
    metaset = pd.read_csv(fpath, dtype={'track_id': object})    # read 'track_id' values as strings
    metaset_x = metaset.track_id.as_matrix()

    # convert from string representation of list to list for each element of the 'genres_all' column
    genres_all_lists = [ast.literal_eval(x) for x in metaset.genres_all.tolist()]
    metaset_y = np.vstack((metaset.genre_top.as_matrix(), genres_all_lists))

    return metaset_x, metaset_y


def get_metadata():
    """
    Public function reads metadata and returns it splitted.

    :return train_x, train_y, valid_x, valid_y, test_x, test_y:
    """
    train_x, train_y = _read_metaset(modeldata_template.format('train.csv'))
    valid_x, valid_y = _read_metaset(modeldata_template.format('valid.csv'))
    test_x, test_y   = _read_metaset(modeldata_template.format('test.csv'))

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def __plot_column_freq(df, index_name):
    """
    Plot values' frequencies for given column.

    :param df:
    :param index_name:
    :return:
    """
    ax = df[index_name].value_counts().plot('bar')
    fig = ax.get_figure()
    fig.savefig(graphs_template.format(index_name + '.png'))


def __extract_id_from_str_list(ids_string, top_ids):
    """
    The idea is to return first top genre that is
    found in the given list of track genres.

    :param ids_string:
    :param top_ids:
    :return:
    """
    for id in ids_string[1:-1].replace(' ', '').split(','):
        if int(id) in top_ids:
            return int(id)
    return None


if __name__ == '__main__':
    """
    Main module that generates train.csv, valid.csv and test.csv.
    It uses :py:class:: DataSize to store only the data specified
    by the dataset size (small, medium, large).
    """
    if len(sys.argv) == 3:
        data_size = DataSize(sys.argv[1])

    genres_df = pd.read_csv(genres_fpath)
    tracks_df = pd.read_csv(tracks_fpath, header=[1], low_memory=False)   # header=[1]: take second level of multi index
    tracks_df = tracks_df.rename(columns={'Unnamed: 0': 'track_id'})      # 'track_id' is originally one level lower

    __plot_column_freq(tracks_df, 'genre_top')

    new_df = tracks_df[['track_id', 'genre_top', 'genres_all', 'split', 'subset']].drop(0)  # select relevant columns
    top_genres = new_df.genre_top.unique()
    top_genres = [genre for genre in top_genres if type(genre) == str]    # get top genres list without illegal elements
    top_genre_ids = [genres_df[genres_df['title'] == genre]['genre_id'].iloc[0] for genre in top_genres]  # to genre ids

    i = 0
    for idx, row in new_df.iterrows():
        genre_id = -1
        # check 'subset' value; if the dataset size we are working with is smaller, skip
        # also, if the track doesn't contain any genre tags, skip as it's useless for this problem
        if not data_size.__ge__(row[4]) or row[2] == '[]':
            new_df.drop(idx, inplace=True)
            continue
        if type(row[1]) != str:             # not all values have 'genre_top' value assigned; fill in using 'genres_all'
            genre_id = __extract_id_from_str_list(row[2], top_genre_ids)
            if genre_id is None:
                print('Note: unable to find top genre tag to assign - removing row')
                new_df.drop(idx, inplace=True)
                continue
        else:                               # replace genre names with corresponding ids
            genre_id = genres_df[genres_df['title'] == row[1]]['genre_id'].iloc[0]
        new_df.loc[idx, 'genre_top'] = genre_id     # replace dataframe values with ids

        # append zeros if track_id is shorted than six characters
        track_id = '0' * (6 - len(row[0])) + row[0]
        if track_id == '065753':
            print(row)
        new_df.loc[idx, 'track_id'] = track_id

        i += 1
        if i % 1000 == 0:
            print('{:.2f}%'.format(i / new_df.shape[0] * 100))  # not 100% accurate as the shape is changing

    new_df = new_df.drop('subset', 1)                   # remove column that is now useless

    train_df = new_df[new_df.split == 'training']
    train_df = train_df.drop('split', 1)    # remove useless columns
    valid_df = new_df[new_df.split == 'validation']
    valid_df = valid_df.drop('split', 1)
    test_df = new_df[new_df.split == 'test']
    test_df = test_df.drop('split', 1)

    train_df.to_csv(modeldata_template.format('train.csv'), encoding='utf-8')
    valid_df.to_csv(modeldata_template.format('valid.csv'), encoding='utf-8')
    test_df.to_csv(modeldata_template.format('test.csv'), encoding='utf-8')
