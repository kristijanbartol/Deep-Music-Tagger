import marshal
import os
import json
import librosa

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.utils.data_utils import get_file


tags_path = os.path.dirname(os.path.abspath(__file__))
ser_data_path = 'dataset.dat'
spectrs_path = 'spectrograms.dat'


class Logger:

    Header = '\033[95m'
    Success = '\033[92m'
    Info = '\033[94m'
    Warning = '\033[93m'
    Error = '\033[91m'
    Bold = '\033[1m'
    Underline = '\033[4m'
    ENDC = '\033[0m'

    def log(self, type, msg):
        print (type + msg + self.ENDC)


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
    # TODO: check how to download
    pass


def generate_spectrograms(serializer, ofile=spectrs_path):
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
if not os.path.isfile(spectrs_path):
    generate_spectrograms(serializer)

# -------------------------------------------------------------------------------------------------------------- #

TH_WEIGHTS_PATH = 'https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/music_tagger_crnn_weights_theano.h5'
TF_WEIGHTS_PATH = 'https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/music_tagger_crnn_weights_tensorflow.h5'


def MusicTaggerCRNN(weights='msd', input_tensor=None,
                    include_top=True):
    '''Instantiate the MusicTaggerCRNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.
    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.
    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        include_top: whether to include the 1 fully-connected
            layer (output layer) at the top of the network.
            If False, the network outputs 32-dim features.
    # Returns
        A Keras model instance.
    '''
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3)(x)
    if include_top:
        x = Dense(50, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)
    if weights is None:
        return model
    else:
        # Load input
        if K.image_dim_ordering() == 'tf':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")

        model.load_weights('data/music_tagger_crnn_weights_%s.h5' % K._BACKEND,
                           by_name=True)
        return model
