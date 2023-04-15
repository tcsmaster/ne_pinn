import torch
import os
from scipy.stats.qmc import Halton, Sobol, scale, LatinHypercube
import numpy as np
from models import *

def save_results(results, directory, file_name):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    extension = '.csv'
    file_path = os.path.join(directory, file_name) + extension
    results.to_csv(file_path)
    print("Data successfully saved!")
    return

def generate_file_name(pde,
                       epochs,
                       hidden_units_1,
                       hidden_units_2,
                       gamma_1,
                       gamma_2,
                       hidden_units_3=None,
                       gamma_3 = None):
    if not hidden_units_3:
        file_name = f"loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}"
    else:
        file_name = f"loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_hidden3_{hidden_units_3}_gamma1_{gamma_1}_gamma2_{gamma_2}_gamma3_{gamma_3}_epochs_{epochs}"
    return file_name

def l2_relative_loss(pred, target):
    return np.linalg.norm(pred- target)/np.linalg.norm(target)
def rmse_vec_error(pred, target):
    return np.linalg.norm(pred - target) / np.sqrt(len(pred))