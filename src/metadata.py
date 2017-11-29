import pandas as pd

fdata_template = '../data/fma_metadata/{}'
graphs_template = '../out/graphs/{}'

genres_fpath = fdata_template.format('genres.csv')
tracks_fpath = fdata_template.format('tracks.csv')


def __plot_column_freq(df, index_name):
    ax = df[index_name].value_counts().plot('bar')
    fig = ax.get_figure()
    fig.savefig(graphs_template.format(index_name + '.png'))


def __extract_id_from_str_list(s):
    return int(s[1:-1].replace(' ', '').split(',')[0])      # expected parameter form is '[e1, e2, ..., ei]'


if __name__ == '__main__':
    genres_df = pd.read_csv(genres_fpath)
    tracks_df = pd.read_csv(tracks_fpath, header=[1], low_memory=False)   # header=[1]: take second level of multi index
    tracks_df = tracks_df.rename(columns={'Unnamed: 0': 'track_id'})      # 'track_id' is originally one level lower

    __plot_column_freq(tracks_df, 'genre_top')

    new_df = tracks_df[['track_id', 'genre_top', 'genres_all', 'split', 'subset']].drop(0)  # select relevant columns

    for idx, row in new_df.iterrows():
        genre_id = -1
        if row[2] == '[]':                  # some tracks do not have any genres assigned; remove them
            new_df.drop(idx, inplace=True)
            continue
        if type(row[1]) != str:             # not all values have 'genre_top' value assigned; fill in using 'genres_all'
            genre_id = __extract_id_from_str_list(row[2])
        else:                               # replace genre names with corresponding ids
            genre_id = genres_df.loc[genres_df['title'] == row[1]]['genre_id'].iloc[0]
        new_df.loc[idx, 'genre_top'] = genre_id

    print(new_df)
