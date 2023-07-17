import logging

# https://towardsdatascience.com/8-advanced-python-logging-features-that-you-shouldnt-miss-a68a5ef1b62d#:~:text=Handler%20specifies%20the%20destination%20of,to%20streams%20such%20as%20sys.
# https://docs.python.org/3/howto/logging-cookbook.html

_log_format = f"[%(levelname)s] - %(name)s - %(filename)s.<%(lineno)d> - %(message)s"
# (%(filename)s).%(funcName)s(%(lineno)d)

def get_file_handler(file_name):
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler

def get_file_handler_meta(file_name):
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(levelname)s] - %(name)s - %(message)s'))
    return file_handler

def get_file_handler_progress(file_name):
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(levelname)s] - %(name)s - %(asctime)s - %(message)s', datefmt="%d-%m-%Y %H:%M:%S"))
    return file_handler

def get_stream_handler():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(logging.Formatter(_log_format))
    return stream_handler

def get_logger(name, file_name, type=None):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if type == 'meta':
        logger.addHandler(get_file_handler_meta(file_name))
    elif type == 'progress':
        logger.addHandler(get_file_handler_progress(file_name))
    else:
        logger.addHandler(get_file_handler(file_name))
    # logger.addHandler(get_stream_handler())
    return logger