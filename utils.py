import os
import numpy as np
from models import *

def save_results(
    content,
    directory,
    file_name
    ):
    """
    Saves a csv file to a specified directory with a given filename.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    extension = '.csv'
    file_path = os.path.join(directory, file_name) + extension
    content.to_csv(file_path)
    print("Data successfully saved!")
    return

def generate_file_name(
    pde,
    epochs,
    hidden_units_1,
    hidden_units_2,
    gamma_1,
    gamma_2
    ):
    """
    Generates a filename for the metrics and model parameter files.
    """
    file_name = f'''loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}'''
    return file_name

def l2_relative_loss(pred, target):
    """
    Calculates the L²-relative error ||pred - target|| / ||target||.
    """
    return np.linalg.norm(pred- target) / np.linalg.norm(target)

def mse_vec_error(pred, target):
    """
    Calculates the mean squared error between the prediction and the target vector
    """
    return np.square(np.linalg.norm(pred - target)) / len(pred)