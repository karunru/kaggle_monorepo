import os
import random
from typing import NoReturn

import cupy as cp
import lightning.pytorch as pl
import numpy as np
import torch


def seed_everything(seed: int = 1234) -> NoReturn:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    cp.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)
