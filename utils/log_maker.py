import os
import threading
import functools

from time import sleep
from datetime import datetime
from numpy import ndarray as np_array

try:
    from termcolor import colored
except ModuleNotFoundError:
    print("Warning: termcolor could not be imported")

    def colored(msg, color):
        return msg


def new_thread(func):
    """Start a new thread for the function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        x = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        x.start()
        x.join()

    return wrapper_timer


class LogMaker(object):
    def __init__(self, config_dict: dict):
        self.log_dir_path = None
        self.log_files = list()
        self.log_id = -1
        self.current_log_id = 0
        self.timezone = config_dict["timezone"]

    def make_log_dir(self, log_dir_path: str):
        self.log_dir_path = log_dir_path
        os.makedirs(self.log_dir_path, exist_ok=True)

    def get_log_id(self):
        self.log_id += 1
        return self.log_id

    def write_log(self, file_name: str, log_message: str, add_date: False):
        if add_date:
            now = datetime.now(self.timezone)
            date = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
        else:
            date = ""

        if file_name in self.log_files:
            log_file_path = os.path.join(self.log_dir_path, file_name)
            with open(log_file_path, 'a') as log_file:
                log_file.write(date + log_message + '\n')
        else:
            self.log_files.append(file_name)
            log_file_path = os.path.join(self.log_dir_path, file_name)
            with open(log_file_path, 'w') as log_file:
                log_file.write(date + log_message + '\n')


log_maker = None


def start_log_maker(config_dict: dict):
    global log_maker
    log_maker = LogMaker(config_dict)


def set_log_dir_path(log_dir_path: str):
    if log_maker is None:
        print(colored("Log maker aren't started yet, so base path will not be set..", color="red"))
        return
    # assert log_maker is not None, "Log maker aren't started yet."
    log_maker.make_log_dir(log_dir_path)


def write_log(file_name: str, log_message, title: str = None, blank_line: bool = True,
              separate_into_lines: bool = True, print_out: bool = False, color: str = None, add_date: bool = False):
    """
    file_name: name of the log file, where the log message belongs
    log_message: the string, list, tuple or string, that is the log message
    title: title of the message, it will be writen with capital before the message
    blank_line: put one blank line after the message
    separate_into_line: if True then list, tuple and dict will be writen more lines
    """
    if log_maker is None:
        return

    if print_out:
        print(colored(log_message, color=color))

    log_id = log_maker.get_log_id()

    while log_id != log_maker.current_log_id:
        sleep(0.1)

    if isinstance(log_message, (str, int, float, complex)) or not separate_into_lines:
        if title is not None:
            log_maker.write_log(file_name, title.upper())
        log_maker.write_log(file_name, str(log_message), add_date)
    elif isinstance(log_message, (list, tuple, np_array)):
        if title is not None:
            log_maker.write_log(file_name, title.upper())
        for message in log_message:
            log_maker.write_log(file_name, str(message), add_date)
    elif isinstance(log_message, dict):
        if title is not None:
            log_maker.write_log(file_name, title.upper())
        for key, value in log_message.items():
            message = str(key) + ': ' + str(value)
            log_maker.write_log(file_name, message, add_date)
    else:
        raise TypeError("Not allow to write {} type into the log!".format(type(log_message)))

    if blank_line:
        log_maker.write_log(file_name, '')

    log_maker.current_log_id += 1

