# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
import os
import sys
from pydoc import locate
import pandas as pd
import scipy as sp
import numpy as np
from scipy.optimize import linear_sum_assignment
import importlib

full_models_a = ["cmids-fullaa40", "cmids-imuaa40"]
full_models_b = ["cmids-fullbb20", "cmids-fullbb20v2", "cmids-imubb20", "cmids-imubb20v2"]
full_models_c = None
imu_models_a = ["cmids-imuaa40"]
imu_models_b = ["cmids-imubb20", "cmids-imubb20v2"]
imu_models_c = None

wa_full = 1.0
wb_full = 1.0
wc_full = 1.0

wa_imu = 1.0
wb_imu = 1.0
wc_imu = 1.0


use_pp = True
use_combined2 = True


device = ["cpu", "cuda"][1]

def get_all_subfolders(folders_list = None, base_path = "../input"):
    if folders_list is None:
        return None
    paths = [f"{base_path}/{folder}" for folder in folders_list]
    folders = [[os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))] for path in paths]
    return [x for xs in folders for x in xs]

def load_experiments(paths):
    if paths is None or len(paths) == 0:
        return []
    exs = []
    for i, ex_path in enumerate(paths):
        sys.path.append(ex_path)

        cfg_module_path = os.path.join(ex_path, "run_cfg.py")
        cfg_spec = importlib.util.spec_from_file_location(f"cfg_{i}", cfg_module_path)
        cfg_module = importlib.util.module_from_spec(cfg_spec)
        cfg_spec.loader.exec_module(cfg_module)
        cfg = cfg_module.cfg
        #cfg = locate(f"run_cfg.cfg")
        
        cfg.device = device
        cfg.num_workers = 0
        cfg.persistent_workers = False
        cfg.dataset.apply_fe = True
        cfg.dataset.fold = -1
    
        #cfg.dataset.df_path = "../input/cmi-detect-behavior-with-sensor-data/train.csv"
        #cfg.dataset.df_demo_path = "../input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
        
        ex = locate("core.experiment.MyExperiment")(cfg)
        ex.load(f"{ex_path}/model/weights")
    
        exs.append(ex)
        sys.path.pop()
        
    return exs

gesture_map = {'Text on phone': 0,
    'Neck - scratch': 1,
    'Eyebrow - pull hair': 2,
    'Forehead - scratch': 3,
    'Forehead - pull hairline': 4,
    'Above ear - pull hair': 5,
    'Neck - pinch skin': 6,
    'Eyelash - pull hair': 7,
    'Cheek - pinch skin': 8,
    'Wave hello': 9,
    'Write name in air': 10,
    'Pull air toward your face': 11,
    'Feel around in tray and pull out an object': 12,
    'Glasses on/off': 13,
    'Drink from bottle/cup': 14,
    'Scratch knee/leg skin': 15,
    'Write name on leg': 16,
    'Pinch knee/leg skin': 17}
gesture_map_inv = {v: k for k, v in gesture_map.items()}



full_models_a_paths = get_all_subfolders(full_models_a)
full_models_a = load_experiments(full_models_a_paths)
full_models_b_paths = get_all_subfolders(full_models_b)
full_models_b = load_experiments(full_models_b_paths)
full_models_c_paths = get_all_subfolders(full_models_c)
full_models_c = load_experiments(full_models_c_paths)
print("Full models a: ",full_models_a_paths)
print(len(full_models_a))
print("Full models b: ",full_models_b_paths)
print(len(full_models_b))
print("Full models c: ",full_models_c_paths)
print(len(full_models_c))

print()

imu_models_a_paths = get_all_subfolders(imu_models_a)
imu_models_a = load_experiments(imu_models_a_paths)
imu_models_b_paths = get_all_subfolders(imu_models_b)
imu_models_b = load_experiments(imu_models_b_paths)
imu_models_c_paths = get_all_subfolders(imu_models_c)
imu_models_c = load_experiments(imu_models_c_paths)
print("IMU models a: ", imu_models_a_paths)
print(len(imu_models_a))
print("IMU models b: ", imu_models_b_paths)
print(len(imu_models_b))
print("IMU models c: ", imu_models_c_paths)
print(len(imu_models_c))

def _greedy_assignment(matrix, nn):
    N, M = matrix.shape
    expanded_matrix = np.tile(matrix, (1, nn))
    _, col_indices = linear_sum_assignment(expanded_matrix)
    original_cols = col_indices % M
    total_cost = sum(matrix[i, original_cols[i]] for i in range(N))
    
    return original_cols.tolist(), total_cost

def fix_nan_inf(x, ones=False):
    if np.isnan(x).any() or np.isinf(x).any():
        if ones:
            x = np.ones_like(x) / len(x)
        else:
            x = np.zeros_like(x)
    return x

class PP3:
    def __init__(self):
        self.dict = {}

    def get_gesture_probs(self, p_comb):
        return p_comb.reshape(4, -1).sum(axis=0).reshape(2, -1).sum(axis=0)
    
    def optimizer(self, mat):
        if mat.shape[0] > mat.shape[1]:
            return self.get_gesture_probs(mat[-1])
        newmat = -np.log(mat + 1e-10)
        sol = _greedy_assignment(newmat, 1)[0][:-1]
        res = mat[-1].copy()
        for i in sol:
            res[i] = 0
        return self.get_gesture_probs(res)
    
    def process_combined(self, subj, pc, raw=False):
        if raw:
            pc = sp.special.softmax(fix_nan_inf(pc))
        else:
            pc = fix_nan_inf(pc, True)
        p_comb = pc.flatten()
        if subj not in self.dict:
            self.dict[subj] = np.array([p_comb])
        else:
            self.dict[subj] = np.concatenate([self.dict[subj], p_comb.reshape(1,-1)], axis=0)
        
        return self.optimizer(self.dict[subj])


pp = PP3()


if use_combined2:
    keys = ["combined2"]
else:
    keys = ["orientation", "gesture"]

def predict(sequence, demographics):
    sequence = sequence.to_pandas()
    demographics = demographics.to_pandas()

    tof_cols = [c for c in sequence.columns if c.startswith("tof_")]
    is_imu_only = sequence[tof_cols].isnull().all(axis=1).all()
    
    if is_imu_only:
        # IMU-only model
        exs_a = imu_models_a
        exs_b = imu_models_b
        exs_c = imu_models_c
        wa = wa_imu
        wb = wb_imu
        wc = wc_imu
    else:
        # full-data model
        exs_a = full_models_a
        exs_b = full_models_b
        exs_c = full_models_c
        wa = wa_full
        wb = wb_full
        wc = wc_full
        
    exs_a[0].config.dataset.df_path = sequence
    exs_a[0].config.dataset.df_demo_path = demographics
    loader_a = exs_a[0].create_loader("test")
    if len(exs_b) > 0:
        exs_b[0].config.dataset.df_path = sequence
        exs_b[0].config.dataset.df_demo_path = demographics
        loader_b = exs_b[0].create_loader("test")
    else:
        loader_b = None
    if len(exs_c) > 0:
        exs_c[0].config.dataset.df_path = sequence
        exs_c[0].config.dataset.df_demo_path = demographics
        loader_c = exs_c[0].create_loader("test")
    else:
        loader_c = None

    res = {}
    for k in keys:
        res[k] = 0.0
    for ex in exs_a:
        pred = ex.run_inference(loader_a)
        for k in keys:
            res[k] += wa * fix_nan_inf(pred[k].cpu().numpy()[0]) / (wa * len(exs_a) + wb * len(exs_b) + wc * len(exs_c))
    for ex in exs_b:
        pred = ex.run_inference(loader_b)
        for k in keys:
            res[k] += wb * fix_nan_inf(pred[k].cpu().numpy()[0]) / (wa * len(exs_a) + wb * len(exs_b) + wc * len(exs_c))
    for ex in exs_c:
        pred = ex.run_inference(loader_c)
        for k in keys:
            res[k] += wc * fix_nan_inf(pred[k].cpu().numpy()[0]) / (wa * len(exs_a) + wb * len(exs_b) + wc * len(exs_c))

    if use_pp:
        if use_combined2:
            ppp = pp.process_combined(sequence["subject"].values[0], res["combined2"], True)
        else:
            ppp = pp.process(sequence["subject"].values[0], res["orientation"], res["gesture"], True)
        label = gesture_map_inv[ppp.argmax()]
    else:
        label = gesture_map_inv[res["gesture"].argmax()]

    return label


import kaggle_evaluation.cmi_inference_server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )



if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print(pd.read_parquet("submission.parquet"))


