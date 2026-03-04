import os
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(__file__))
TODAY = datetime.strftime(datetime.today(), "%Y%m%d")
YESTDAY =  datetime.strftime(datetime.today() - timedelta(1), "%Y%m%d")
THREE_MOTHS_AGO = datetime.strftime(datetime.today() - timedelta(30*3), "%Y%m%d")
CPUS = os.cpu_count()
TIMEGAP = 2