import pandas as pd
import sys

fdata_template = '../data/fma_metadata/{}'
modeldata_template = '../out/model_dataset/{}'
graphs_template = '../out/graphs/{}'

genres_fpath = fdata_template.format('genres.csv')
tracks_fpath = fdata_template.format('tracks.csv')


class DataSize:

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


def __plot_column_freq(df, index_name):
    ax = df[index_name].value_counts().plot('bar')
    fig = ax.get_figure()
    fig.savefig(graphs_template.format(index_name + '.png'))


def __extract_id_from_str_list(s):
    return int(s[1:-1].replace(' ', '').split(',')[0])      # expected parameter form is '[e1, e2, ..., ei]'


if __name__ == '__main__':
    if len(sys.argv) == 3:
        data_size = DataSize(sys.argv[1])

    genres_df = pd.read_csv(genres_fpath)
    tracks_df = pd.read_csv(tracks_fpath, header=[1], low_memory=False)   # header=[1]: take second level of multi index
    tracks_df = tracks_df.rename(columns={'Unnamed: 0': 'track_id'})      # 'track_id' is originally one level lower

    __plot_column_freq(tracks_df, 'genre_top')

    new_df = tracks_df[['track_id', 'genre_top', 'genres_all', 'split', 'subset']].drop(0)  # select relevant columns

    for idx, row in new_df.iterrows():
        genre_id = -1
        # check 'subset' value; if the dataset size we are working with is smaller, skip
        # also, if the track doesn't contain any genre tags, skip as it's useless for this problem
        if not data_size.__ge__(row[4]) or row[2] == '[]':
            new_df.drop(idx, inplace=True)
            continue
        if type(row[1]) != str:             # not all values have 'genre_top' value assigned; fill in using 'genres_all'
            genre_id = __extract_id_from_str_list(row[2])
        else:                               # replace genre names with corresponding ids
            genre_id = genres_df.loc[genres_df['title'] == row[1]]['genre_id'].iloc[0]
        new_df.loc[idx, 'genre_top'] = genre_id

    new_df = new_df.drop('subset', 1)

    train_df = new_df[new_df.split == 'training']
    train_df = train_df.drop('split', 1)
    valid_df = new_df[new_df.split == 'validation']
    valid_df = valid_df.drop('split', 1)
    test_df = new_df[new_df.split == 'test']
    test_df = test_df.drop('split', 1)

    train_df.to_csv(modeldata_template.format('train.csv'), encoding='utf-8')
    valid_df.to_csv(modeldata_template.format('valid.csv'), encoding='utf-8')
    test_df.to_csv(modeldata_template.format('test.csv'), encoding='utf-8')
