import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Burgers_process import *
from utils import *


pde = "Burgers"
epochs=1000
hidden_units_1 = 100
hidden_units_2 = 100

gamma_1_list = [0.5, 0.6, 0.7, 0.8, 0.9]
gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9]

rmse_error = np.zeros((5, 5), dtype=object)
rel_l2_error = np.zeros((5, 5), dtype=object)
data = np.load("Burgers.npz")
t, x, usol = data["t"], data["x"], data["usol"]

for gamma_1 in gamma_1_list:
    test_data = torch.stack(torch.meshgrid(torch.tensor([x], dtype=torch.float32).squeeze(),
                                           torch.tensor([t], dtype=torch.float32).squeeze(),
                                           indexing="ij")).reshape(2, -1).T
    true_sol = usol.reshape(-1, 1)
    for gamma_2 in gamma_2_list:
        net = BurgersNet(MLP2(num_input=2,
                                  num_output=1,
                                  hidden_units_1=hidden_units_1,
                                  hidden_units_2=hidden_units_2,
                                  gamma_1=gamma_1,
                                  gamma_2=gamma_2
                             ),
                             device=torch.device('cpu')
                  )
        path = os.getcwd()+ f"/results/{pde}/2layer/normalized/loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}_model.pth"
        net.model.load_state_dict(torch.load(path,map_location='cpu'))
        net.model.eval()
        with torch.no_grad():
                pred = net.model(test_data)
        pred = pred.detach().numpy()
        rmse_error[gamma_2_list.index(gamma_2), gamma_1_list.index(gamma_1)] = rmse_vec_error(pred, true_sol)
        rel_l2_error[gamma_2_list.index(gamma_2), gamma_1_list.index(gamma_1)] = l2_relative_loss(pred, true_sol)

err_dir = f"/content/thesis/error_tables/{pde}/2layer"
if not os.path.isdir(err_dir):
    os.makedirs(err_dir)
pd.DataFrame(rmse_error, index = ["gamma_2 = 0.5", "gamma_2 = 0.6", "gamma_2 = 0.7", "gamma_2 = 0.8", "gamma_2 = 0.9"], columns = ["gamma_1 = 0.5", "gamma_1 = 0.6", "gamma_1 = 0.7", "gamma_1 = 0.8", "gamma_1 = 0.9"]).to_csv(err_dir + "/Poisson_2layer_rmse_table.csv")
pd.DataFrame(rel_l2_error, index = ["gamma_2 = 0.5", "gamma_2 = 0.6", "gamma_2 = 0.7", "gamma_2 = 0.8", "gamma_2 = 0.9"], columns = ["gamma_1 = 0.5", "gamma_1 = 0.6", "gamma_1 = 0.7", "gamma_1 = 0.8", "gamma_1 = 0.9"]).to_csv(err_dir + "/Poisson_2layer_rel_l2_table.csv")