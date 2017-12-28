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
from utility import plot_training_progress

logs_path = '../out/logs/train.log'

batch_size = 6 
img_height = 96
img_width = 1366
channels = 1

num_epochs = 1

# Decaying by factor of ~0.91 after each epoch (for batch_size 6)
lr_starting = 8e-4
lr_decay = 0.9999714

start_time = time.time()
logger = Logger(batch_size, num_epochs, start_time)
plot_data = dict()
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['f1_score'] = []
plot_data['lr'] = []


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


def batched_evaluate(dataset, iter_limit):
    print(dataset.get_number_of_batches(batch_size))
    print(iter_limit)
    loss = 0
    # Limit number of evaluated batches
    iter_range = range(dataset.get_number_of_batches(batch_size))
    if iter_limit is not None:
        iter_range = range(iter_limit)
    for _ in iter_range:
        batch_x, batch_y = dataset.next_batch(batch_size)
        loss += model.test_on_batch(batch_x.reshape(-1, img_height, img_width, channels), batch_y)
    return loss


def log_score(data, iter_limit=None):
    logger.color_print(logger.Info, '\n-------\nEvaluating {} score...'.format(data.dataset_label))
    op_start_time = time.time()
    # Using batches to evaluate as it's intensive to load the whole set at once
    evaluated = batched_evaluate(data, iter_limit)
    loss = evaluated / data.get_number_of_batches(batch_size) if iter_limit is None else evaluated / iter_limit

    op_time, h, m, s = get_times(op_start_time, start_time)
    logger.color_print(logger.Info, 'epoch {0} | {1}_loss: {2:.2f} | {3:.2f}s | {4:02d}:{5:02d}:{6:02d}\n-------\n'
                       .format(epoch + 1, data.dataset_label, loss, op_time, h, m, s))
    if data.dataset_label is not 'test':
        plot_data['{}_loss'.format(data.dataset_label)] += [loss]


data = input_data.get_data()

model = build_model(data.train.get_output_size())

adam = Adam(lr=lr_starting, decay=lr_decay)

print('Compiling model...')
model.compile(loss='categorical_crossentropy', optimizer=adam)

for epoch in range(num_epochs):
    data.train.shuffle()
    number_of_batches = data.test.get_number_of_batches(batch_size)
    for i in range(number_of_batches):
        op_start_time = time.time()
        batch_x, batch_y = data.test.next_batch(batch_size)
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

    # Approximate train log score with ~1/4 dataset size for efficiency
    #log_score(data.train, iter_limit=data.train.get_number_of_batches(batch_size) // 4)
    #log_score(data.valid)
    current_lr = lr_starting * (lr_decay ** data.train.get_number_of_batches(batch_size)) ** (epoch + 1)
    logger.color_print(logger.Info, 'Current learning rate: {}'.format(current_lr))
    plot_data['train_loss'] = 0
    plot_data['valid_loss'] = 0
    plot_data['lr'] += [current_lr]
    plot_data['f1_score'] = 0   # TODO
    #plot_training_progress(plot_data)
    logger.dump(logs_path)

log_score(data.test)
