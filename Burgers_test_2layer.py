import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Burgers_process import *
from utils import *

def l2_relative_loss(pred, target):
    return np.linalg.norm(pred- target)/np.linalg.norm(target)
def rmse_vec_error(pred, target):
    return np.linalg.norm(pred - target) / np.sqrt(len(pred))

pde = "Burgers"
epochs=20000
hidden_units_1 = 100
hidden_units_2 = 100

gamma_1_list = [0.5, 0.7, 1.0]
gamma_2_list = [0.5, 0.7, 1.0]

rmse_error = np.zeros((4, 3, 3), dtype=object)
rel_l2_error = np.zeros((4, 3, 3), dtype=object)

t_space = [0, 0.25, 0.5, 0.75]
data = np.load("Burgers.npz")
t, x, usol = data["t"], data["x"], data["usol"]

for gamma_1 in gamma_1_list:
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize = (20, 10), dpi=300)
    for ax, time in zip(axs.ravel(), t_space):
        test_data = torch.stack(torch.meshgrid(torch.tensor([time], dtype=torch.float32),
                                               torch.tensor([x], dtype=torch.float32).squeeze(),
                                               indexing="ij")).reshape(2, -1).T
        true_sol = usol[:, np.where(t == time)[0]]
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
            path = os.getcwd()+ f"/results/{pde}/2layer/normalized/Adam_with_amsgrad/loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}_model.pth"
            net.model.load_state_dict(torch.load(path,map_location='cpu'))
            net.model.eval()
            with torch.no_grad():
                pred = net.model(test_data)
            pred = pred.detach().numpy()
            rmse_error[t_space.index(time), gamma_2_list.index(gamma_2), gamma_1_list.index(gamma_1)] = rmse_vec_error(pred, true_sol)
            rel_l2_error[t_space.index(time), gamma_2_list.index(gamma_2), gamma_1_list.index(gamma_1)] = l2_relative_loss(pred, true_sol)
            ax.plot(x, pred, label = f"$\gamma_2 = {{{gamma_2}}}$")
        ax.plot(x, true_sol, label="True solution")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid()
        ax.legend(loc="best")
        ax.set_title(f"t={time}")
    plt.suptitle(f"Prediction for $\gamma_1 = {{{gamma_1}}}$")
    file_name = f"plot_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}"
    plt.savefig("/content/thesis/figures_test/Burgers_test/2layer/" + file_name + ".jpg")
out_rmse = np.hstack(rmse_error)
out_rel_l2 = np.hstack(rel_l2_error)
pd.DataFrame(out_rmse, index = ["gamma_2 = 0.5", "gamma_2 = 0.7", "gamma_2 = 1.0"], columns=pd.MultiIndex.from_product((["t=0", "t=0.25", "t=0.5", "t=0.75"], ["gamma_1 = 0.5", "gamma_1 = 0.7", "gamma_1 = 1.0"]))).to_csv("/content/thesis/Error_tables/Burgers/Burgers_2layer_rmse_table.csv")
pd.DataFrame(out_rel_l2, index = ["gamma_2 = 0.5", "gamma_2 = 0.7", "gamma_2 = 1.0"], columns=pd.MultiIndex.from_product((["t=0", "t=0.25", "t=0.5", "t=0.75"], ["gamma_1 = 0.5", "gamma_1 = 0.7", "gamma_1 = 1.0"]))).to_csv("/content/thesis/Error_tables/Burgers/Burgers_2layer_rel_l2_table.csv")

