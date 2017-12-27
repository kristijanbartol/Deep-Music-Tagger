class Logger:

    Header = '\033[95m'
    Success = '\033[92m'
    Info = '\033[94m'
    Warning = '\033[93m'
    Error = '\033[91m'
    Bold = '\033[1m'
    Underline = '\033[4m'
    ENDC = '\033[0m'

    log_lines = []

    def color_print(self, type, msg):
        print (type + msg + self.ENDC)
        self._add_log_line(msg)

    def _add_log_line(self, line):
        self.log_lines.append(line)

    def dump(self, fpath):
        with open(fpath, 'w') as dump_file:
            for line in self.log_lines:
                dump_file.write(line)

