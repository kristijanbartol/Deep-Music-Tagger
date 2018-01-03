from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, ELU, GRU
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Reshape
from keras.optimizers import SGD, Adam

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

batch_size = 3 
img_height = 96
img_width = 1366
channels = 1

num_epochs = 500

# Decaying by factor of ~0.91 after each epoch (for batch_size 6)
lr_starting = 9e-4
lr_decay = 0.9999714

start_time = time.time()
logger = Logger(batch_size, num_epochs, lr_starting)
score_data = dict()
score_data['train_loss'] = []
score_data['valid_loss'] = []
score_data['f1_score'] = []
score_data['lr'] = []

optimizers = {'sgd': SGD(lr=0.001, momentum=0.9, nesterov=True),
              'adam': Adam(lr=lr_starting, decay=lr_decay)}


def build_model(output_size):
    channel_axis = 3
    freq_axis = 1
    padding = 37

    input_shape = (img_height, img_width, channels)
    print('Building model...')

    # TODO: try only a dense layer alone to learn (fix dimensions error)

    model = Sequential()
    model.add(ZeroPadding2D(padding=(0, padding), data_format='channels_last', input_shape=input_shape))
    model.add(BatchNormalization(axis=freq_axis, name='bn_0_freq'))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv1'))
    model.add(BatchNormalization(axis=channel_axis, name='bn1'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))
    #model.add(Dropout(0.1, name='dropout1'))

    model.add(Conv2D(128, (3, 3), padding='same', name='conv2'))
    model.add(BatchNormalization(axis=channel_axis, name='bn2'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2'))
    #model.add(Dropout(0.1, name='dropout2'))

    model.add(Conv2D(128, (3, 3), padding='same', name='conv3'))
    model.add(BatchNormalization(axis=channel_axis, name='bn3'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3'))
    #model.add(Dropout(0.1, name='dropout3'))

    model.add(Conv2D(128, (3, 3), padding='same', name='conv4'))
    model.add(BatchNormalization(axis=channel_axis, name='bn4'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4'))
    #model.add(Dropout(0.1, name='dropout4'))

    model.add(Reshape(target_shape=(15, 128)))

    model.add(GRU(32, return_sequences=True, name='gru1'))
    model.add(GRU(32, return_sequences=False, name='gru2'))

    #model.add(Dropout(0.3, name='dropout_final'))

    model.add(Dense(output_size, activation='softmax', name='output', input_shape=input_shape))

    return model


def batched_evaluate(dataset, batches):
    loss = 0
    acc = 0
    for _ in range(batches):
        batch_x, batch_y = dataset.next_batch(batch_size, mode='silent')
        loss_, acc_ = model.test_on_batch(batch_x.reshape(-1, img_height, img_width, channels), batch_y)
        loss += loss_
        acc += acc_
    return loss, acc


def evaluate(data, batches):
    logger.color_print(logger.Bold, '\n-------\nEvaluating {} score ({} batches)...'.format(data.dataset_label, batches))
    op_start_time = time.time()
    # Using batches to evaluate as it's intensive to load the whole set at once
    loss, acc = batched_evaluate(data, batches)
    loss /= batches
    acc /= batches

    op_time, h, m, s = get_times(op_start_time, start_time)
    logger_level = logger.Bold
    if data.dataset_label is not 'test':
        score_data['{}_loss'.format(data.dataset_label)] += [loss]
    else:
        logger_level = logger.Success
    logger.color_print(logger_level,
                      '\n-------\nepoch {} | {}_loss: {:.2f} | acc: {:.2f} | {:.2f}s | {:02d}:{:02d}:{:02d}\n-------\n'
                       .format(epoch + 1, data.dataset_label, loss, acc, op_time, h, m, s))
    logger.dump(session_path)


def get_lr(batch):
    return lr_starting * (lr_decay ** batch) ** (epoch + 1)


data = data.get_data()

model = build_model(data.number_of_classes)

opt_name = 'sgd'
optimizer = optimizers[opt_name]

print('Compiling model...')
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[metrics.categorical_accuracy])
print('Model metrics: {}'.format(model.metrics_names))

data.train.limit_dataset_size(30)
data.valid.limit_dataset_size(6)
for epoch in range(num_epochs):
    #data.train.shuffle()
    number_of_batches = data.train.get_number_of_batches(batch_size)
    for i in range(number_of_batches):
        op_start_time = time.time()
        batch_x, batch_y = data.train.next_batch(batch_size)
        model.train_on_batch(batch_x.reshape(-1, img_height, img_width, channels), batch_y)

        # Log (log :)) loss, current position and times
        op_time, h, m, s = get_times(op_start_time, start_time)
        #if (i + 1) % 50 == 0:
        loss, acc = model.evaluate(batch_x.reshape(-1, img_height, img_width, channels), batch_y, batch_size)
        lr = get_lr(i + 1)
        logger.color_print(logger.Bold,
                           'epoch {} | batch {}/{} | loss: {:.2f} | acc: {} | lr: {:.4E} | {:.2f}s | {:02d}:{:02d}:{:02d}'
                           .format(epoch + 1, i + 1, number_of_batches, loss, acc, lr, op_time, h, m, s))
        #else:
        #    print('epoch {} | batch {}/{} | {:.2f}s | {:02d}:{:02d}:{:02d}'
        #          .format(epoch + 1, i + 1, number_of_batches, op_time, h, m, s))

    # Approximate train log score with ~1/4 dataset size for efficiency
    evaluate(data.train, data.train.get_number_of_batches(batch_size))
    evaluate(data.valid, data.valid.get_number_of_batches(batch_size))

    score_data['lr'] += [get_lr(data.train.get_number_of_batches(batch_size))]
    #score_data['f1_score'] += [f1_score(...)]

    #scores_path = scores_template.format(epoch, batch_size, opt_name, lr_starting, lr_decay)
    #print('Saving scores (train/valid loss, lr and f1-score) to {}'.format(scores_path))
    #save_scores(score_data, scores_path)
    #save_model_path = save_model_template.format(epoch, batch_size, opt_name, lr_starting, lr_decay)
    #print('Saving current model state with all parameters to {}'.format(save_model_path))
    #model.save(save_model_path)
    #save_indices(data.train.indices, indices_template.format(epoch, batch_size, opt_name, lr_starting, lr_decay))

evaluate(data.test, data.test.get_number_of_batches(batch_size))
