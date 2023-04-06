import torch
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Poisson_process import *
from utils import *

def l2_relative_loss(input, target):
    return np.linalg.norm(input- target)/np.linalg.norm(target)
def rmse_vec_error(input, target):
    return np.linalg.norm(input - target) / np.sqrt(len(input))

pde = "Poisson"
device = torch.device("cpu")
gamma_1_list = [0.5, 0.7, 1.0]
gamma_2_list = [0.5, 0.7, 1.0]
gamma_3_list = [0.5, 0.7, 1.0]
hidden_units_1 = 100
hidden_units_2 = 100
hidden_units_3 = 100
epochs=20000

rmse_error = np.zeros((3, 3, 3), dtype=object)
rel_l2_error = np.zeros((3, 3, 3), dtype=object)
for el in itertools.product(gamma_2_list, gamma_3_list):
    fig= plt.figure(figsize=(20, 10), dpi=300)
    ax = fig.add_subplot(1,1,1)
    test_data = torch.linspace(-1, 1, 30).reshape(1, -1).T
    true_sol = torch.sin(np.pi*test_data).detach().numpy()
    for gamma_1_ind, gamma_1 in enumerate(gamma_1_list):
        net = PoissonNet(MLP3(num_input=1, num_output=1, hidden_units_1=hidden_units_1, hidden_units_2 = hidden_units_2, hidden_units_3 = hidden_units_3, gamma_1 = gamma_1, gamma_2 = el[0], gamma_3 = el[1]), device=device)
        path = os.getcwd()+ f"/results/{pde}/3layer/normalized/Adam_with_amsgrad/loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_hidden3_{hidden_units_3}_gamma1_{gamma_1}_gamma2_{el[0]}_gamma3_{el[1]}_epochs_{epochs}_model.pth"
        net.model.load_state_dict(torch.load(path,map_location='cpu'))
        net.model.eval()
        with torch.no_grad():
            pred = net.model(test_data)
        pred = pred.detach().numpy()
        rmse_error[gamma_2_list.index(el[0]), gamma_3_list.index(el[1]), gamma_1_ind] = rmse_vec_error(pred, true_sol)
        rel_l2_error[gamma_2_list.index(el[0]), gamma_3_list.index(el[1]), gamma_1_ind] = l2_relative_loss(pred, true_sol)
        ax.plot(test_data, pred, label=f"$\gamma_1 = {{{gamma_1}}}$")
    ax.plot(test_data, true_sol, label = "True solution")
    ax.legend(loc="best")
    ax.grid()
    ax.set_title(f"Prediction for $\gamma_2 = {{{el[0]}}}, \gamma_3 = {{{el[1]}}}$")
    file_name = f"plot_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_hidden3_{hidden_units_3}_gamma1_{gamma_1}_gamma2_{el[0]}_gamma3_{el[1]}_epochs_{epochs}"
    plt.savefig("/content/thesis/figures_test/Poisson_test/3layer/" + file_name + ".jpg")

out_rmse = np.row_stack(rmse_error.reshape(3, -1))
out_rel_l2 = np.row_stack(rel_l2_error.reshape(3, -1))
pd.DataFrame(out_rmse, index = ["gamma_1 = 0.5", "gamma_1 = 0.7", "gamma_1 = 1.0"], columns=pd.MultiIndex.from_product((["gamma_2 = 0.5", "gamma_2 = 0.7", "gamma_2 = 1.0"], ["gamma_3 = 0.5", "gamma_3 = 0.7", "gamma_3 = 1.0"]))).to_csv("/content/thesis/Error_tables/Poisson/rmse_table.csv")
pd.DataFrame(out_rel_l2, index = ["gamma_1 = 0.5", "gamma_1 = 0.7", "gamma_1 = 1.0"], columns=pd.MultiIndex.from_product((["gamma_2 = 0.5", "gamma_2 = 0.7", "gamma_2 = 1.0"], ["gamma_3 = 0.5", "gamma_3 = 0.7", "gamma_3 = 1.0"]))).to_csv("/content/thesis/Error_tables/Poisson/rel_l2_table.csv")