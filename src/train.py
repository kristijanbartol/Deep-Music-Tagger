from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, ELU, GRU
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Reshape
from keras.optimizers import SGD

import numpy as np
import math
import time

import input_data
from melspec import get_times

batch_size = 5
img_height = 96
img_width = 1366
channels = 1

num_epochs = 30


def build_model(output_size):
    channel_axis = 3
    freq_axis = 1
    padding = 37

    input_shape = (img_height, img_width, channels)
    print('Building model...')

    model = Sequential()
    model.add(ZeroPadding2D(padding=(0, padding), data_format='channels_last', input_shape=input_shape))
    model.add(BatchNormalization(axis=freq_axis, name='bn_0_freq'))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv1'))
    model.add(BatchNormalization(axis=channel_axis, name='bn1'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))
    model.add(Dropout(0.1, name='dropout1'))

    model.add(Conv2D(128, (3, 3), padding='same', name='conv2'))
    model.add(BatchNormalization(axis=channel_axis, name='bn2'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2'))
    model.add(Dropout(0.1, name='dropout2'))

    model.add(Conv2D(128, (3, 3), padding='same', name='conv3'))
    model.add(BatchNormalization(axis=channel_axis, name='bn3'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3'))
    model.add(Dropout(0.1, name='dropout3'))

    model.add(Conv2D(128, (3, 3), padding='same', name='conv4'))
    model.add(BatchNormalization(axis=channel_axis, name='bn4'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4'))
    model.add(Dropout(0.1, name='dropout4'))

    model.add(Reshape(target_shape=(15, 128)))

    model.add(GRU(32, return_sequences=True, name='gru1'))
    model.add(GRU(32, return_sequences=False, name='gru2'))

    model.add(Dropout(0.3, name='dropout_final'))

    model.add(Dense(output_size, activation='softmax', name='output'))

    return model


def multi_output_cross_entropy(labels, outputs):
    return 1 / outputs * (np.sum(labels * K.log(outputs)))


data = input_data.get_data()

model = build_model(data.train.get_output_size())

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
print('Compiling model...')
model.compile(loss=multi_output_cross_entropy, optimizer=sgd)

start_time = time.time()

for epoch in range(num_epochs):
    number_of_batches = int(math.ceil(data.train.get_dataset_size() / batch_size))
    data.train.shuffle()
    for i in range(number_of_batches):
        op_start_time = time.time()
        batch_x, batch_y = data.train.next_batch(batch_size)
        # import pdb
        # pdb.set_trace()
        model.train_on_batch(batch_x.reshape(-1, img_height, img_width, channels), batch_y)

        # Log current position and times
        op_time, overall_h, overall_m, overall_s = get_times(op_start_time, start_time)
        current_time = time.time()
        print('epoch {0} | batch {1} / {2} | {3:.2f}s | {4:02d}:{5:02d}:{6:02d}'
              .format(epoch + 1, i + 1, number_of_batches, op_time, overall_h, overall_m, overall_s))

x_test, y_test = data.test.all_loaded()
score = model.evaluate(x_test, y_test, batch_size=batch_size)
