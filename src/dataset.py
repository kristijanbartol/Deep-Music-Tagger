import pandas

fpaths_template = '/home/kristijan/FER/projekt/data/fma_metadata/{}'

genres_fpath = fpaths_template.format('genres.csv')
tracks_fpath = fpaths_template.format('tracks.csv')


def get_genres_by_track_id(track_id):
    pass


if __name__ == '__main__':
    genres_data = pandas.read_csv(genres_fpath)
    tracks_data = pandas.read_csv(tracks_fpath)

    # TODO: map track_ids to genres and store in a separate file (JSON or CSV)
