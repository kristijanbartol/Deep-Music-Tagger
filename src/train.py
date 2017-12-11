from keras.models import Sequential
from keras.layers import Dense, Dropout, ELU, GRU, Activation
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Reshape
from keras.optimizers import SGD

import input_data

channel_axis = 3
freq_axis = 1

data = input_data.get_data()
x_train, y_train = data.train.all_loaded()
x_test, y_test = data.test.all_loaded()

model = Sequential()

# input: 96x1366 images with grayscale channel
model.add(ZeroPadding2D(padding=(0, 37), input_shape=(96, 1366, 1)))
model.add(BatchNormalization(axis=freq_axis, name='bn_0_freq'))

# Conv block 1
model.add(Conv2D(64, 3, 3, border_mode='same', name='conv1'))
model.add(BatchNormalization(axis=channel_axis, mode=0, name='bn1'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))
model.add(Dropout(0.1, name='dropout1'))

# Conv block 2
model.add(Conv2D(128, 3, 3, border_mode='same', name='conv2'))
model.add(BatchNormalization(axis=channel_axis, mode=0, name='bn2'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2'))
model.add(Dropout(0.1, name='dropout2'))

# Conv block 3
model.add(Conv2D(128, 3, 3, border_mode='same', name='conv3'))
model.add(BatchNormalization(axis=channel_axis, mode=0, name='bn3'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3'))
model.add(Dropout(0.1, name='dropout3'))

# Conv block 4
model.add(Conv2D(128, 3, 3, border_mode='same', name='conv4'))
model.add(BatchNormalization(axis=channel_axis, mode=0, name='bn4'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4'))
model.add(Dropout(0.1, name='dropout4'))

# prepare dimensions for GRUs
model.add(Reshape((15, 128)))

# GRU block 1
model.add(GRU(32, return_sequences=True, name='gru1'))

# GRU block 2
model.add(GRU(32, return_sequences=False, name='gru2'))
model.add(Dropout(0.3))

# Dense layer
model.add(Dense(50, name='logits'))

# apply softmax to get probabilistic outputs
model.add(Activation('softmax', name='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
