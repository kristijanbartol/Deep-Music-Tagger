from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, ELU, GRU
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Reshape
from keras.optimizers import SGD

import numpy as np

import input_data

batch_size = 50
img_height = 96
img_width = 1366
channels = 1


def build_model(input_tensor, output_size):
    channel_axis = 3
    freq_axis = 1
    padding = 37

    # Convert tensor to keras tensor
    if not K.is_keras_tensor(input_tensor):
        input_tensor = Input(tensor=input_tensor, shape=(batch_size, img_height, img_width, channels))

    # TODO: (temporal fix)  After Input function call, I ran into a bug with 'None' dimension prepended
    input_tensor._keras_shape = [dim for dim in input_tensor._keras_shape if dim is not None]

    # Input block
    x = ZeroPadding2D(padding=(0, padding), data_format='channels_last')(input_tensor)
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Conv2D(64, (3, 3), padding='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)

    # (50, 1, 15, 128) -> (50, 15, 128)
    # Note: target_shape does not include the batch axis.
    x = Reshape(target_shape=(15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3)(x)

    x = Dense(output_size, activation='softmax', name='output')(x)

    return Model(input_tensor, x)


def multi_output_cross_entropy(labels, outputs):
    return 1 / outputs.shape[0].value * (np.sum(labels * K.log(outputs)))


data = input_data.get_data()
# x_train, y_train = data.train.all_loaded()
# x_test, y_test = data.test.all_loaded()

model = build_model(K.placeholder(shape=(batch_size, img_height, img_width, channels)), data.train.get_output_size())

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=multi_output_cross_entropy, optimizer=sgd)

model.fit(data.train.next_batch(batch_size), data.train.labels, batch_size=batch_size, epochs=1)
score = model.evaluate(data.test.next_batch(batch_size), data.test.labels, batch_size=batch_size)
