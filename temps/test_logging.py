import os
import logging
from deeplens.utils import set_logger

iter = 5

for i in range(iter):
    
    dir_name = f'./Dir{i+1}'
    os.makedirs(dir_name)
    message = f'This text for the {i+1} time logging.'
    
    set_logger(dir_name)
    logging.info(message)
    
    
    # log_string = f'output{i+1}.log'
    # log_name = './temps/' + log_string + '.log'
    # message = f'This text for the {i+1} time logging.'
    
    # # set root
    # root = logging.getLogger()
    # root.setLevel(logging.INFO)
    
    # # remove old file handlers
    # root.handlers.clear()
    
    # # attach new file handlers
    # BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    # DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    # formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    # fhlr = logging.FileHandler(log_name)
    # fhlr.setFormatter(formatter)
    # fhlr.setLevel(logging.INFO)
    # root.addHandler(fhlr)

    # logging.info(message)