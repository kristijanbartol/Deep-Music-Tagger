class Logger:

    Header = '\033[95m'
    Success = '\033[92m'
    Info = '\033[94m'
    Warning = '\033[93m'
    Error = '\033[91m'
    Bold = '\033[1m'
    Underline = '\033[4m'
    ENDC = '\033[0m'

    def __init__(self, batch_size, epochs, start_time):
        first_line = 'Batch size: {} | Epochs: {} | Started: {}\n'.format(batch_size, epochs, start_time)
        self.log_lines = [first_line]

    def color_print(self, type, msg):
        print (type + msg + self.ENDC)
        self._add_log_line(msg)

    def _add_log_line(self, line):
        self.log_lines.append(line)

    def dump(self, fpath):
        with open(fpath, 'w') as dump_file:
            for line in self.log_lines:
                dump_file.write(line + '\n')
        self.log_lines = []


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(16, 8))

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
    ax2.set_title('Learning rate')
    ax2.plot(x_data, data['lr'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='learning_rate')
    ax2.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.pdf')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)

