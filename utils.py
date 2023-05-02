import os
import numpy as np
from models import *

def save_results(
    content,
    directory,
    file_name
    ):
    """
    Saves the content in the specified directory. Content could be DataFrame or figure
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if isinstance(content, pd.DataFrame):
        extension = '.csv'
        file_path = os.path.join(directory, file_name) + extension
        content.to_csv(file_path)
    elif isinstance(content, matplotlib.figure.Figure):
        extension = '.jpg'
        file_path = os.path.join(directory, file_name) + extension
        content.savefig(file_path, bbox_inches="tight", dpi=300)
    else:
        raise ValueError("Implementation for this type of content is not implemented yet!")
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
    Generates a filename for the test loss and model parameter files.
    """
    file_name = f'''loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}'''
    return file_name

def l2_relative_loss(pred, target):
    """
    Calculates the L²-relative error ||pred - target|| / ||target||.
    """
    return np.linalg.norm(pred- target) / np.linalg.norm(target)

def mse_vec_error(pred, target):
    return np.linalg.norm(pred - target) / np.sqrt(len(pred))