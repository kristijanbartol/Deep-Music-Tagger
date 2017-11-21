import json
import marshal
import os

import librosa

from src.utility import Logger

# TODO: update main module
##############################################

DIGITAL7_API_KEY = ''
try:
    DIGITAL7_API_KEY = os.environ['DIGITAL7_API_KEY']
except KeyError:
    DIGITAL7_API_KEY = None

# downloaded song tags dataset files path (https://labrosa.ee.columbia.edu/millionsong/lastfm)
tags_path = os.path.dirname(os.path.abspath(__file__))
# serialized tags output file path
ser_data_path = 'dataset.dat'
# generated spectrograms output file path
spectrograms_path = 'spectrograms.dat'


class Serializer:

    num_of_batches = 0
    BATCH_LIMIT = 1
    params_path = 'params.conf'

    loading_in_progress = False

    def save_params(self):
        params = {
            self.BATCH_LIMIT.__str__() : self.BATCH_LIMIT,
            self.num_of_batches.__str__() : self.num_of_batches
        }

        with open('params.conf', 'w') as f_params:
            json.dump(params, f_params)

    @staticmethod
    def load_params(self):
        return json.load('params.conf')

    def dump(self, rootdir=tags_path, ofile=ser_data_path):
        ouf = open(ofile, 'wb')
        data = {}
        batch_size = 0

        for subdir, _, files in os.walk(rootdir):
            for file in files:
                if file[len(file) - 5:] == '.json':
                    with open(os.path.join(subdir, file)) as data_file:
                        content = json.load(data_file)
                        data[content['track_id']] = content['tags']
                        batch_size += 1
                if batch_size >= self.BATCH_LIMIT:
                    marshal.dump(data, ouf)
                    data = {}
                    batch_size = 0
                    self.num_of_batches += 1

        if data is not {}:
            marshal.dump(data, ouf)
            self.num_of_batches += 1
        ouf.close()
        self.save_params()

    @staticmethod
    def merge_two_dicts(x, y):
        z = x.copy()
        z.update(y)
        return z

    '''
    Returns pair: (data, status of operation)
    '''
    def load_batch(self, ifile=ser_data_path):
        inf = open(ifile, 'rb')

        data = {}
        try:
            data = self.merge_two_dicts(data, marshal.load(inf))
        except Exception as e:
            logger.log(logger.Error, str(e) + '. Delete generated file and try again.')
            return data, False
        inf.close()

        return data, True


def download_raw_audio_batch(data):
    audio_batch = {}

    for trackid in data:
        url = 'http://api.7digital.com/1.2/track/preview?redirect=false'
        url += '&trackid=' + str(trackid)
        url += '&oauth_consumer_key=' + DIGITAL7_API_KEY
        xmldoc = url_call(url)
        status = xmldoc.getAttribute('status')
        if status != 'ok':
            return ''
        url_element = xmldoc.getElementsByTagName('url')[0]
        preview = url_element.firstChild.nodeValue
        print (preview)

        # TODO: get audio here
        audio_batch[trackid] = ''

    return audio_batch


def generate_spectrograms(serializer, ofile=spectrograms_path):
    while True:
        spectr_grams = {}
        data, status = serializer.load_batch()
        if status is False:
            break

        raw_audio_batch = download_raw_audio_batch(data)
        for d in data:
            spectr_grams[d] = librosa.feature.melspectrogram(y=raw_audio_batch, sr=22050, hop_length=512, n_mfcc=13)

        ouf = open(ofile, 'wb')
        marshal.dump(spectr_grams, ouf)


logger = Logger()
serializer = Serializer()
if not os.path.isfile(ser_data_path):
    serializer.dump()
else:
    logger.log(logger.Info, 'Found generated dataset file. If you want to regenerate new, remove the file manually.')

# NOTE: you will never actually save raw audio; convert it to spectrograms right away to preserve GBs of disk space
if not os.path.isfile(spectrograms_path):
    generate_spectrograms(serializer)

if __name__ == '__main__()':
    pass
