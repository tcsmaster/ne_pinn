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
                       sampler=None,
                       hidden_units_3=None,
                       gamma_3 = None):
    if not sampler:
        if not hidden_units_3:
            file_name = f"loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}"
        else:
            file_name = f"loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_hidden3_{hidden_units_3}_gamma1_{gamma_1}_gamma2_{gamma_2}_gamma3_{gamma_3}_epochs_{epochs}"
    else:
      if not hidden_units_3:
            file_name = f"loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}_sampler_{sampler}"
      else:
            file_name = f"loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_hidden3_{hidden_units_3}_gamma1_{gamma_1}_gamma2_{gamma_2}_gamma3_{gamma_3}_epochs_{epochs}_sampler_{sampler}"
    return file_name

def data_gen(space, n_samples, sampler):
    if len(space)==1:
        if sampler == "LHS":
            sampler = LatinHypercube(d=len(space))
            return torch.Tensor(scale(sampler.random(n=n_samples), l_bounds=space[0][0], u_bounds=space[0][1])).float()
        elif sampler == "Halton":
            sampler = Halton(d=len(space))
            return torch.tensor(scale(sampler.random(n=n_samples), l_bounds=space[0][0], u_bounds=space[0][1])).float()
        elif sampler == "Sobol":
            sampler = Sobol(d=len(space))
            return torch.Tensor(scale(sampler.random(n=n_samples), l_bounds=space[0][0], u_bounds=space[0][1])).float()  #TODO: look into pape
        elif sampler== 'random':
            one_dim_points = np.random.default_rng().uniform(size=(n_samples, len(space)))
            return torch.Tensor((space[0][1] - space[0][0])*one_dim_points + space[0][0]).float()
        else:
            raise ValueError('This sampling method has not been implemented yet!')
    else:
        if sampler == "LHS":
            sampler = LatinHypercube(d=len(space[0]))
            return torch.Tensor(scale(sampler.random(n=n_samples), l_bounds=space[0], u_bounds=space[1])).float()
        elif sampler == "Halton":
            sampler = Halton(d=len(space[0]))
            return torch.tensor(scale(sampler.random(n=n_samples), l_bounds=space[0], u_bounds=space[1])).float()
        elif sampler == "Sobol":
            sampler = Sobol(d=len(space[0]))
            return torch.Tensor(scale(sampler.random(n=n_samples), l_bounds=space[0], u_bounds=space[1])).float()  #TODO: look into paper
        elif sampler== 'random':
            one_dim_points = np.random.default_rng().uniform(size=(n_samples, len(space)))
            return torch.Tensor((space[1] - space[0])*one_dim_points + space[0]).float()
        else:
            raise ValueError('This sampling method has not been implemented yet!')

def u_func(x):
    return (
        -(
            torch.exp(x[:, 1:2]) * torch.sin(x[:, 2:3]+ x[:, 3:4])
            + torch.exp(x[:, 3:4]) * torch.cos(x[:, 1:2] + x[:, 2:3])
        )
        * torch.exp(-x[:, 0:1])
    )


def v_func(x):
    return (
        -(
            torch.exp(x[:, 2:3]) * torch.sin(x[:, 3:4] +  x[:, 1:2])
            + torch.exp(x[:, 1:2]) * torch.cos(x[:, 2:3] +  x[:, 3:4])
        )
        * torch.exp(-x[:, 0:1])
    )


def w_func(x):
    return (
        -(
            torch.exp(x[:, 3:4]) * torch.sin(x[:, 1:2] +  x[:, 2:3])
            + torch.exp(x[:, 2:3]) * torch.cos(x[:, 3:4] +  x[:, 1:2])
        )
        * torch.exp(-x[:, 0:1])
    )


def p_func(x):
    return (
        -0.5
        * (
            torch.exp(2 * x[:, 1:2])
            + torch.exp(2 * x[:, 2:3])
            + torch.exp(2 * x[:, 3:4])
            + 2
            * torch.sin(x[:, 1:2] +  x[:, 2:3])
            * torch.cos(x[:, 3:4] +  x[:, 1:2])
            * torch.exp((x[:, 2:3] + x[:, 3:4]))
            + 2
            * torch.sin(x[:, 2:3] +  x[:, 3:4])
            * torch.cos(x[:, 1:2] +  x[:, 2:3])
            * torch.exp((x[:, 3:4] + x[:, 1:2]))
            + 2
            * torch.sin(x[:, 3:4] +  x[:, 1:2])
            * torch.cos(x[:, 2:3] +  x[:, 3:4])
            * torch.exp((x[:, 1:2] + x[:, 2:3]))
        )
        * torch.exp(-2 * x[:, 0:1])
    )

