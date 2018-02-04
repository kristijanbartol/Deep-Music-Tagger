from keras import backend as K
from keras import metrics
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, ELU, GRU
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Reshape
from keras.optimizers import SGD, Adam

import tensorflow as tf

from sklearn.metrics import f1_score

import time

import data
from melspec import get_times
from utility import Logger
from utility import plot_training_progress, save_scores

session_path = '../../out/logs/session.log'
save_model_template = '../../out/models/crowdai/model_{}_{}_{}_{}_{}.h5'
scores_template = '../../out/scores/crowdai/scores_{}_{}_{}_{}_{}.out'
indices_template = '../../out/models/crowdai/indices_{}_{}_{}_{}_{}.out'

batch_size = 10
img_height = 96
img_width = 1366
channels = 1

num_epochs = 500

# Decaying by factor of ~0.91 after each epoch (for batch_size 6)
lr_starting = 5e-3
lr_decay = 0.999

start_time = time.time()
logger = Logger(batch_size, num_epochs, lr_starting)
score_data = dict()
score_data['train_loss'] = []
score_data['valid_loss'] = []
score_data['f1_score'] = []
score_data['lr'] = []

optimizers = {'sgd': SGD(lr=0.001, momentum=0.9, nesterov=True),
              'adam': Adam(lr=lr_starting, decay=lr_decay)}

data = data.get_data()

model = Sequential()

limit = 200

data.train.limit_dataset_size(limit)

# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
input_shape = (img_height, img_width, channels)
model.add(Reshape(target_shape=(img_height * img_width,), input_shape=input_shape))
model.add(Dense(data.number_of_classes, activation='softmax', name='output', input_shape=input_shape))

#model.add(ZeroPadding2D(padding=(0, padding), data_format='channels_last', input_shape=input_shape))
#model.add(BatchNormalization(axis=freq_axis, name='bn_0_freq'))

#model.add(Conv2D(64, (3, 3), padding='same', name='conv1'))
#model.add(BatchNormalization(axis=channel_axis, name='bn1'))
#model.add(ELU())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))
#model.add(Dropout(0.1, name='dropout1'))

#model.add(Conv2D(128, (3, 3), padding='same', name='conv2'))
#model.add(BatchNormalization(axis=channel_axis, name='bn2'))
#model.add(ELU())
#model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2'))
#model.add(Dropout(0.1, name='dropout2'))

#model.add(Conv2D(128, (3, 3), padding='same', name='conv3'))
#model.add(BatchNormalization(axis=channel_axis, name='bn3'))
#model.add(ELU())
#model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3'))
#model.add(Dropout(0.1, name='dropout3'))

#model.add(Conv2D(128, (3, 3), padding='same', name='conv4'))
#model.add(BatchNormalization(axis=channel_axis, name='bn4'))
#model.add(ELU())
#model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4'))
#model.add(Dropout(0.1, name='dropout4'))

#model.add(Reshape(target_shape=(15, 128)))

#model.add(GRU(32, return_sequences=True, name='gru1'))
#model.add(GRU(32, return_sequences=False, name='gru2'))

#model.add(Dropout(0.3, name='dropout_final'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=lr_starting, decay=lr_decay)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

for epoch in range(10):
    batch_x, batch_y = data.train.next_batch(batch_size)
    model.train_on_batch(batch_x.reshape(-1, img_height, img_width, channels), batch_y)

test_x, test_y = data.test.next_batch(limit)

score = model.evaluate(test_x.reshape(-1, img_height, img_width, channels), test_y, batch_size=limit)
print(score)
