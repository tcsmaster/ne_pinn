import torch
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Poisson_process import *
from utils import *
def l2_relative_loss(pred, target):
    return np.linalg.norm(pred- target)/np.linalg.norm(target)
def rmse_vec_error(pred, target):
    return np.linalg.norm(pred - target) / np.sqrt(len(pred))

pde = "Poisson"
epochs=1000
hidden_units_1 = 100
hidden_units_2 = 100

gamma_1_list = [0.5, 0.6, 0.7, 0.8, 0.9]
gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9]

rmse_error = np.zeros((5, 5), dtype=object)
rel_l2_error = np.zeros((5, 5), dtype=object)

test_data = torch.linspace(-1, 1, 30).reshape(1, -1).T
true_sol = torch.sin(np.pi*test_data).detach().numpy()
for gamma_1 in gamma_1_list:
    net = PoissonNet(MLP3(num_input=1, num_output=1, hidden_units_1=hidden_units_1, hidden_units_2 = hidden_units_2, hidden_units_3 = hidden_units_3, gamma_1 = gamma_1, gamma_2 = el[0], gamma_3 = el[1]), device=device)
    path = os.getcwd()+ f"/results/{pde}/3layer/normalized/loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_hidden3_{hidden_units_3}_gamma1_{gamma_1}_gamma2_{el[0]}_gamma3_{el[1]}_epochs_{epochs}_model.pth"
    net.model.load_state_dict(torch.load(path,map_location='cpu'))
    net.model.eval()
    with torch.no_grad():
        pred = net.model(test_data)
    pred = pred.detach().numpy()
    rmse_error[gamma_2_list.index(el[0]), gamma_3_list.index(el[1]), gamma_1_ind] = rmse_vec_error(pred, true_sol)
    rel_l2_error[gamma_2_list.index(el[0]), gamma_3_list.index(el[1]), gamma_1_ind] = l2_relative_loss(pred, true_sol)

err_dir = "/content/thesis/Error_tables/Poisson/"
if not os.path.isdir(err_dir):
    os.makedirs(err_dir)
pd.DataFrame(rmse_error, index = ["gamma_2 = 0.5", "gamma_2 = 0.6", "gamma_2 = 0.7", "gamma_2 = 0.8", "gamma_2 = 0.9"], columns = ["gamma_1 = 0.5", "gamma_1 = 0.6", "gamma_1 = 0.7", "gamma_1 = 0.8", "gamma_1 = 0.9"]).to_csv(err_dir + "Poisson_2layer_rmse_table.csv")
pd.DataFrame(rel_l2_error, index = ["gamma_2 = 0.5","gamma_2 = 0.6", "gamma_2 = 0.7", "gamma_2 = 0.8", "gamma_2 = 0.9"], columns = ["gamma_1 = 0.5", "gamma_1 = 0.6", "gamma_1 = 0.7", "gamma_1 = 0.8", "gamma_1 = 0.9"]).to_csv(err_dir + "Poisson_2layer_rel_l2_table.csv")