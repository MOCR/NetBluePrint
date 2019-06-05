"""

	PrintProgress by Arnaud de Broissia

"""

import sys
import time


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[1;36m'
    OKGREEN = '\033[1;32m'
    WARNING = '\033[93m'
    FAIL = '\033[1;31m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class void_printer:
    def __init__(self, message=None, printInterval=1):
        pass

    def __call__(self, message):
        return self

    def __enter__(self, timer=False):
        return self

    def __exit__(self, type, value, traceback):
        if all((type, value, traceback)):
            raise type, value, traceback


class printProg:
    """
        Context for clean printing of state of the system.
    """

    def __init__(self, message, printInterval=1):
        self.message = [message]
        self.printInterval = printInterval
        self.lastPrint = 0
        self.status = [""]
        self.level_count = -1
        self.on_line = False
        self.timers=[-1]

    def __call__(self, message, timer=False):
        self.message.append(message)
        self.status.append("")
        if timer:
            self.timers.append(1)
        else:
            self.timers.append(-1)
        return self

    def __enter__(self):
        self.level_count += 1
        self.status[-1] = bcolors.OKBLUE + "..." + bcolors.ENDC
        if self.timers[-1] == 1:
            self.timers[-1] = time.time()
        if self.on_line:
            sys.stdout.write("\n")
            self.on_line = False
        self.printStatus()
        return self

    def __exit__(self, type, value, traceback):
        if all((type, value, traceback)):
            self.status[-1] = bcolors.FAIL + "FAILED" + bcolors.ENDC
            self.printError(str(type))
            self.printStatus(True)
            self.level_count -= 1
            del self.status[-1]
            del self.message[-1]
            raise type, value, traceback
        else:
            self.status[-1] = bcolors.OKGREEN + "OK" + bcolors.ENDC
            if self.timers[-1] != -1:
                self.status[-1] +=bcolors.OKGREEN + " in " +  "{0:.2f}".format(time.time() - self.timers[-1]) + "s"+ bcolors.ENDC
            self.printStatus(True)
            self.level_count -= 1
            del self.status[-1]
            del self.timers[-1]
            del self.message[-1]

    def printInferior(self, f1, f2):
        if f1 < f2 and self.lastPrint % self.printInterval == 0:
            if f2 != 0:
                self.status[-1] = bcolors.OKBLUE + str(float(f1) / float(f2) * 100.0)[:4] + bcolors.ENDC
            else:
                self.status[-1] = bcolors.OKBLUE + "Err" + bcolors.ENDC
            self.printStatus()
        return f1 < f2

    def printWarning(self, message):
        if self.on_line:
            sys.stdout.write("\n")
            self.on_line = False
        self.indent_message()
        sys.stdout.write("[" + bcolors.WARNING + "Warning" + bcolors.ENDC + "] " + message + "\033[K\n")
        sys.stdout.flush()
        self.printStatus()

    def printError(self, message):
        if self.on_line:
            sys.stdout.write("\n")
            self.on_line = False
        self.indent_message()
        sys.stdout.write("[" + bcolors.FAIL + "ERROR" + bcolors.ENDC + "] " + message + "\033[K\n")
        sys.stdout.flush()
        self.printStatus()

    def printResult(self, flag, message):
        if self.on_line:
            sys.stdout.write("\n")
            self.on_line = False
        self.indent_message()
        sys.stdout.write("[" + bcolors.OKBLUE + flag + bcolors.ENDC + "] " + message + "\033[K\n")
        sys.stdout.flush()
        self.printStatus()

    def printStatus(self, retour=False):
        self.indent_message()
        sys.stdout.write(self.message[-1] + " [")
        sys.stdout.write(self.status[-1] + "]\033[K")
        if retour:
            sys.stdout.write("\n")
            self.on_line = False
        else:
            sys.stdout.write("\r")
            self.on_line = True
        sys.stdout.flush()

    def setStatus(self, status, printInterval=False):
        self.status[-1] = bcolors.OKBLUE + status + bcolors.ENDC
        if not printInterval or self.lastPrint % self.printInterval == 0:
            self.printStatus()
        if printInterval:
            self.lastPrint += 1

    def indent_message(self):
        for i in range(self.level_count):
            sys.stdout.write("-")


def printWarning(message):
    sys.stdout.write("[" + bcolors.WARNING + "Warning" + bcolors.ENDC + "] " + message + "\n")
    sys.stdout.flush()
