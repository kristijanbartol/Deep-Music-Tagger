import tensorflow as tf
import numpy as np

import input_data
import model

config = dict()
config['max_epochs'] = 8
config['batch_size'] = 50
config['weight_decay'] = 1e-2
config['lr_policy'] = {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}

img_height = 96
img_width = 1366


def train(sess, train_set, train_f, loss_f, config):
    batch_size = config['batch_size']
    max_epoch = config['max_epochs']

    num_examples = train_set.images.shape[0]
    num_batches = num_examples // batch_size

    for epoch in range(1, max_epoch + 1):
        train_set.shuffle()
        for i in range(num_batches):
            batch = train_set.next_batch(batch_size)
            sess.run(train_f, feed_dict={x: batch[0], y: batch[1]})
            if i % 5 == 0:
                loss = sess.run(loss_f, feed_dict={x: batch[0], y: batch[1]})
                print('epoch %d/%d, step %d/%d, batch loss = %.2f'
                      % (epoch, max_epoch, i * batch_size, num_examples, loss))
            # if i > 0 and i % 50 == 0:
            #    accuracy = sess.run(accuracy_f, feed_dict={x: batch[0], y: batch[1]})
            #    print('Train accuracy = %.2f' % accuracy)


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

    # with tf.name_scope('accuracy'):
    #    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    #    correct_prediction = tf.cast(correct_prediction, tf.float32)
    # accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        train(session, data.train, train_op, cross_entropy_loss, config)

        # print('test accuracy {}'.format(accuracy.eval(feed_dict={x: data.test.images, y: data.test.labels})))
