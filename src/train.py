from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, ELU, GRU
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Reshape
from keras.optimizers import SGD, Adam

import tensorflow as tf
import numpy as np
import time

import input_data
from melspec import get_times
from utility import Logger

logs_template = '../out/logs/{}'

batch_size = 6 
img_height = 96
img_width = 1366
channels = 1

num_epochs = 1


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


def batched_evaluate(dataset, batch_size=batch_size):
    loss = 0
    for _ in range(dataset.get_number_of_batches(batch_size)):
        batch_x, batch_y = dataset.next_batch(batch_size)
        loss += model.evaluate(batch_x.reshape(-1, img_height, img_width, channels), batch_y, batch_size)
    return loss


data = input_data.get_data()

model = build_model(data.train.get_output_size())

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=8e-4, decay=0.91)

print('Compiling model...')
model.compile(loss='categorical_crossentropy', optimizer=adam)

start_time = time.time()
logger = Logger()

for epoch in range(num_epochs):
    data.train.shuffle()
    number_of_batches = data.train.get_number_of_batches(batch_size)
    for i in range(number_of_batches):
        op_start_time = time.time()
        batch_x, batch_y = data.train.next_batch(batch_size)
        model.train_on_batch(batch_x.reshape(-1, img_height, img_width, channels), batch_y)

        # Log (log :)) loss, current position and times
        op_time, overall_h, overall_m, overall_s = get_times(op_start_time, start_time)
        if (i + 1) % 50 == 0:
            loss = model.evaluate(batch_x.reshape(-1, img_height, img_width, channels), batch_y, batch_size)
            logger.color_print(logger.Info,
                    'epoch {0} | batch {1} / {2} | loss: {3:.2f} | {4:.2f}s | {5:02d}:{6:02d}:{7:02d}'
                    .format(epoch + 1, i + 1, number_of_batches, loss, op_time, overall_h, overall_m, overall_s))
        else:
            print('epoch {0} | batch {1} / {2} | {3:.2f}s | {4:02d}:{5:02d}:{6:02d}'
                    .format(epoch + 1, i + 1, number_of_batches, op_time, overall_h, overall_m, overall_s))

    logger.color_print(logger.Info, '\n-------\nEvaluating validation score...')
    # Using batches because it's intensive to load the whole validation set
    op_start_time = time.time()
    valid_loss = batched_evaluate(data.valid)
    op_time, h, m,s = get_times(op_start_time, start_time)
    logger.color_print(logger.Info, 
            'epoch {0} | valid_loss: {1:.2f} | {2:.2f}s | {3:02d}:{4:02d}:{5:02d}\n-------\n'
            .format(epoch + 1, valid_loss / data.valid.get_number_of_batches(batch_size), op_time, h, m, s))

logger.color_print(logger.Info, '\n\n-------\n\nEvaluating test score...')
op_start_time = time.time()
test_loss = batched_evaluate(data.test)
op_time, h, m,s = get_times(op_start_time, start_time)
logger.color_print(logger.Success, '\n-------\ntest_loss: {0:.2f} | {1:.2f}s | {2:02d}:{3:02d}:{4:02d}\n-------\n'
        .format(test_loss / data.test.get_number_of_batches(batch_size), op_time, h, m, s))

logger.dump(logs_template.format('train.log'))

