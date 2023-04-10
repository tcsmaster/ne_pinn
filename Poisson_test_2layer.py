import torch
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
device = torch.device("cpu")
gamma_1_list = [0.5, 0.6, 0.7, 0.8, 0.9]
gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9]
hidden_units_1 = 100
hidden_units_2 = 100
epochs=2000

rmse_error = np.zeros((5, 5), dtype=object)
rel_l2_error = np.zeros((5, 5), dtype=object)
for gamma_1 in gamma_1_list:
    fig= plt.figure(figsize=(20, 10), dpi=300)
    ax = fig.add_subplot(1,1,1)
    test_data = torch.linspace(-1, 1, 30).reshape(1, -1).T
    true_sol = torch.sin(np.pi*test_data).detach().numpy()
    for gamma_2 in gamma_2_list:
        net = PoissonNet(MLP2(num_input=1, num_output=1, hidden_units_1=hidden_units_1, hidden_units_2 = hidden_units_2, gamma_1 = gamma_1, gamma_2 = gamma_2), device=device)
        path = os.getcwd()+ f"/results/{pde}/2layer/normalized/loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}_model.pth"
        net.model.load_state_dict(torch.load(path,map_location='cpu'))
        net.model.eval()
        with torch.no_grad():
            pred = net.model(test_data)
        pred = pred.detach().numpy()
        rmse_error[gamma_2_list.index(gamma_2), gamma_1_list.index(gamma_1)] = rmse_vec_error(pred.ravel(), true_sol.ravel())
        rel_l2_error[gamma_2_list.index(gamma_2), gamma_1_list.index(gamma_1)] = l2_relative_loss(pred, true_sol)
        ax.plot(test_data, pred, label=f"$\gamma_2 = {{{gamma_2}}}$")
    ax.plot(test_data, true_sol, label = "True solution")
    ax.legend(loc="best")
    ax.grid()
    ax.set_title(f"Prediction for $\gamma_1 = {{{gamma_1}}}$")
    file_name = f"plot_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}"
    fig_dir= "/content/thesis/figures_test/Poisson_test/2layer/"
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_dir + file_name + ".jpg")
err_dir= "/content/thesis/Error_tables/Poisson/"
if not os.path.isdir(err_dir):
        os.makedirs(err_dir)
pd.DataFrame(rmse_error, index = ["gamma_2 = 0.5", "gamma_2 = 0.6","gamma_2 = 0.7", "gamma_2 = 0.8","gamma_2 = 0.9"], columns=["gamma_1 = 0.5", "gamma_2 = 0.6","gamma_1 = 0.7", "gamma_2 = 0.8","gamma_1 = 0.9"]).to_csv(err_dir + f"rmse_table_epochs_{epochs}.csv")
pd.DataFrame(rel_l2_error, index = ["gamma_2 = 0.5", "gamma_2 = 0.6","gamma_2 = 0.7", "gamma_2 = 0.8","gamma_2 = 0.9"], columns=["gamma_1 = 0.5", "gamma_2 = 0.6","gamma_1 = 0.7","gamma_2 = 0.8", "gamma_1 = 0.9"]).to_csv(err_dir + f"rel_l2_table_epochs_{epochs}.csv")