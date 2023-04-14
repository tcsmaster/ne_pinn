import torch
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from Burgers_process import *
from utils import *

def l2_relative_loss(pred, target):
    return np.linalg.norm(pred- target)/np.linalg.norm(target)
def rmse_vec_error(pred, target):
    return np.linalg.norm(pred - target) / np.sqrt(len(pred))

pde = "Burgers"
epochs=1000
hidden_units_1 = 100
hidden_units_2 = 100
hidden_units_3 = 100

gamma_1_list = [0.5, 0.7, 0.9]
gamma_2_list = [0.5, 0.7, 0.9]
gamma_3_list = [0.5, 0.7, 0.9]
rmse_error = np.zeros((3, 3, 3), dtype=object)
rel_l2_error = np.zeros((3, 3, 3), dtype=object)
data = np.load("Burgers.npz")
t, x, usol = data["t"], data["x"], data["usol"]

for el in itertools.product(gamma_2_list, gamma_3_list):
    test_data = torch.stack(torch.meshgrid(torch.tensor([x], dtype=torch.float32).squeeze(),
                                           torch.tensor([t], dtype=torch.float32).squeeze(),
                                           indexing="ij")).reshape(2, -1).T
    true_sol = usol.reshape(-1, 1)
    for gamma_1 in gamma_1_list:
        net = BurgersNet(MLP3(num_input=2,
                                  num_output=1,
                                  hidden_units_1=hidden_units_1,
                                  hidden_units_2=hidden_units_2,
                                  hidden_units_3=hidden_units_3,
                                  gamma_1 = gamma_1,
                                  gamma_2 = el[0],
                                  gamma_3 = el[1]
                             ),
                             device=torch.device('cpu')
                  )
        path = os.getcwd()+ f"/results/{pde}/3layer/normalized/SGD/loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_hidden3_{hidden_units_3}_gamma1_{gamma_1}_gamma2_{el[0]}_gamma3_{el[1]}_epochs_{epochs}_model.pth"
        net.model.load_state_dict(torch.load(path,map_location='cpu'))
        net.model.eval()
        with torch.no_grad():
                pred = net.model(test_data)
        pred = pred.detach().numpy()
        rmse_error[gamma_2_list.index(el[0]), gamma_3_list.index(el[1]), gamma_1_list.index(gamma_1)] = rmse_vec_error(pred, true_sol)
        rel_l2_error[gamma_2_list.index(el[0]), gamma_3_list.index(el[1]), gamma_1_list.index(gamma_1)] = l2_relative_loss(pred, true_sol)
out_rmse = np.row_stack(rmse_error.reshape(3, -1))
out_rel_l2 = np.row_stack(rel_l2_error.reshape(3, -1))
err_dir = f"/content/thesis/error_tables/{pde}/3layer/"
if not os.path.isdir(err_dir):
    os.makedirs(err_dir)
pd.DataFrame(out_rmse, index = ["gamma_2 = 0.5", "gamma_2 = 0.7", "gamma_2 = 0.9"], columns =pd.MultiIndex.from_product((["gamma_2 = 0.5", "gamma_2 = 0.7", "gamma_2 = 0.9"], ["gamma_3 = 0.5", "gamma_3 = 0.7", "gamma_3 = 0.9"]))).to_csv(err_dir + f"{pde}_3layer_rmse_table.csv")
pd.DataFrame(out_rel_l2, index = ["gamma_2 = 0.5", "gamma_2 = 0.7", "gamma_2 = 0.9"], columns=pd.MultiIndex.from_product((["gamma_2 = 0.5", "gamma_2 = 0.7", "gamma_2 = 0.9"], ["gamma_3 = 0.5", "gamma_3 = 0.7", "gamma_3 = 0.9"]))).to_csv(err_dir + f"{pde}_3layer_rel_l2_table.csv")