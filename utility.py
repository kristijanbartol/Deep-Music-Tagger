class Logger:

    Header = '\033[95m'
    Success = '\033[92m'
    Info = '\033[94m'
    Warning = '\033[93m'
    Error = '\033[91m'
    Bold = '\033[1m'
    Underline = '\033[4m'
    ENDC = '\033[0m'

    def log(self, type, msg):
        print (type + msg + self.ENDC)