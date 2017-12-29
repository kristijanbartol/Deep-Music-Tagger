import matplotlib.pyplot as plt
import numpy as np
import os

import time

train_plt_path = '../out/graphs/training.pdf'


class Logger:
    """
    Class Logger gives a nice way to highlight important messages on runtime.
    """
    
    Header = '\033[95m'
    Success = '\033[92m'
    Info = '\033[94m'
    Warning = '\033[93m'
    Error = '\033[91m'
    Bold = '\033[1m'
    Underline = '\033[4m'
    ENDC = '\033[0m'

    def __init__(self, batch_size, epochs):
        """
        Creates instance of Logger, but also creates and keeps first
        log line containing general session information.

        :param batch_size:
        :param epochs:
        :return:
        """
        first_line = 'Batch size: {} | Epochs: {} | Started: {}\n'.format(batch_size, epochs, time.strftime("%H:%M:%S"))
        self.log_lines = [first_line]

    def color_print(self, type, msg):
        """
        Prints messages in color.
        """
        print (type + msg + self.ENDC)
        self._add_log_line(msg)

    def _add_log_line(self, line):
        """
        Private method that adds printed to line the list of logged lines.

        :param line:
        :return:
        """
        self.log_lines.append(line)

    def dump(self, dump_path):
        """
        Writes all the logged lines to file with given path and empties list.
        """
        with open(dump_path, 'w') as dump_file:
            for line in self.log_lines:
                dump_file.write(line + '\n')
        self.log_lines = []


def save_scores(data, scores_path):
    """
    Saves train loss, valid loss, f1-score and current learning rate
    to a file on provided path. This information can be used to make
    plots afterwards or to determine the optimal epoch to make prediction.

    :param data: dictionary with known keys
    :param scores_path:
    :return:
    """
    with open(scores_path, 'a') as fscores:
        for i in range(len(data['train_loss'])):
            fscores.write('{} | {} | {} | {} | {}\n'
                          .format(i, data['train_loss'][i], data['valid_loss'][i], data['f1_score'][i], data['lr'][i]))


def plot_training_progress(data):
    """
    This doesn't work for me when I run the whole thing in terminal,
    but can be used separately to plot data saved with :func:`save_scores`

    :param data: dictionary with known keys
    :return:
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('F1-score')
    ax2.plot(x_data, data['f1_score'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='f1_score')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(train_plt_path, 'training_plot.pdf')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)


def save_prediction(data, predict, prediction_path):
    with open(prediction_path) as fprediction:
        fprediction.write(str(predict(data)))
