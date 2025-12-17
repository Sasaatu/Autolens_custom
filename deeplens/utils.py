import os
import random 
import numpy as np
import torch
import logging

def gpu_init(gpu=0):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("Using: {}".format(device))
    torch.set_default_tensor_type('torch.FloatTensor')
    return device

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def set_logger(dir='./'):
    
    log_name = f"{dir}/output.log"
    
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    # remove old handlers
    root.handlers.clear()
    
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    # attach new stream handler
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel(logging.INFO)
    root.addHandler(chlr)

    # attach new file handler
    fhlr = logging.FileHandler(log_name)
    fhlr.setFormatter(formatter)
    fhlr.setLevel(logging.INFO)
    root.addHandler(fhlr)