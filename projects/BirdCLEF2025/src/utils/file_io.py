# https://github.com/go5paopao/paogle/blob/master/paogle/utils/file_io.py
import os
import pickle
import sys

import torch
from tqdm import tqdm

from .timer import timer


def save_model(save_name, epoch, loss, acc, model, optimizer):
    state = {
        "epoch": epoch,
        "loss": loss,
        "acc": acc,
        "weight": model.state_dict(),
        "optimizer": optimizer.state_dict()["param_groups"],
    }
    torch.save(state, save_name)


def save_pickle(obj, file_path):
    with timer(f"save {file_path}"):
        max_bytes = 2**31 - 1
        bytes_out = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        n_bytes = sys.getsizeof(bytes_out)
        with open(file_path, "wb") as f_out:
            for idx in tqdm(range(0, n_bytes, max_bytes)):
                f_out.write(bytes_out[idx : idx + max_bytes])


def load_pickle(file_path):
    with timer(f"load {file_path}"):
        max_bytes = 2**31 - 1
        input_size = os.path.getsize(file_path)
        bytes_in = bytearray(0)
        with open(file_path, "rb") as f_in:
            for _ in tqdm(range(0, input_size, max_bytes)):
                bytes_in += f_in.read(max_bytes)
        with timer("pickle loads"):
            obj = pickle.loads(bytes_in)
    return obj
