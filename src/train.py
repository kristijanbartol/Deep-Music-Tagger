import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

import input_data
import model

config = dict()
config['max_epochs'] = 8
config['batch_size'] = 50
config['weight_decay'] = 1e-2
# TODO: add lr_policy for gradient descent optimizer
config['lr_policy'] = {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}

img_height = 96
img_width = 1366


def build_model(inputs, num_classes, config):
    weight_decay = config['weight_decay']
    conv1sz = 16
    conv2sz = 32
    fc1sz = 512

    with tf.name_scope('reshape'):
        inputs = tf.reshape(inputs, [-1, 28, 28, 1])

    with tf.contrib.framework.arg_scope([layers.convolution2d],
                                        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.convolution2d(inputs, conv1sz,
                                   weights_initializer=layers.variance_scaling_initializer(), scope='conv1')

        with tf.contrib.framework.arg_scope([layers.max_pool2d],
                                            kernel_size=5, stride=1, padding='SAME'):
            net = layers.max_pool2d(net, scope='max_pool1')
            net = layers.convolution2d(net, conv2sz, scope='conv2')
            net = layers.max_pool2d(net, scope='max_pool2')

    with tf.contrib.framework.arg_scope([layers.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.flatten(net, scope='flatten')
        net = layers.fully_connected(net, fc1sz, scope='fc1')
        logits_ = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')

    return logits_


def train(sess, train_set, accuracy_f, train_f, loss_f, config):
    batch_size = config['batch_size']
    max_epoch = config['max_epochs']

    num_examples = train_set.images.shape[0]
    num_batches = num_examples // batch_size

    for epoch in range(1, max_epoch + 1):
        # TODO: how to permute the dataset after each epoch?
        for i in range(num_batches):
            batch = train_set.next_batch(batch_size)
            sess.run(train_f, feed_dict={x: batch[0], y: batch[1]})
            if i % 5 == 0:
                loss = sess.run(loss_f, feed_dict={x: batch[0], y: batch[1]})
                print('epoch %d/%d, step %d/%d, batch loss = %.2f'
                      % (epoch, max_epoch, i * batch_size, num_examples, loss))
            if i % 100 == 0:
                conv1_var = tf.contrib.framework.get_variables('conv1/weights:0')[0]
                conv1_weights = conv1_var.eval(session=sess)
            if i > 0 and i % 50 == 0:
                accuracy = sess.run(accuracy_f, feed_dict={x: batch[0], y: batch[1]})
                print('Train accuracy = %.2f' % accuracy)


def multi_output_cross_entropy(labels, outputs):
    loss_sum = 0
    for idx, output in enumerate(outputs):
        loss_sum += - labels[idx] * np.log(output)
    return loss_sum / outputs.shape[0]


if __name__ == '__main__':

    data = input_data.get_data()
    y_length = data.train.y.shape[0]

    x = tf.placeholder(tf.float32, [None, img_width * img_height])
    y = tf.placeholder(tf.float32, [None, data.train.y.shape[0]])

    outputs = model.build_model(input_tensor=x)

    with tf.name_scope('loss'):
        cross_entropy_loss = multi_output_cross_entropy(labels=y, outputs=outputs)

    with tf.name_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        train(session, mnist.train, accuracy, train_op, cross_entropy_loss, config)

        print('test accuracy {}'.format(accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})))

