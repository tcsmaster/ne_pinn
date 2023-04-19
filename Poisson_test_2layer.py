import torch
import pandas as pd
import matplotlib.pyplot as plt
from Poisson_process import PoissonNet
from utils import *

plt.rcParams.update({
    "font.monospace": [],
    "figure.figsize": (12,8),
    "axes.labelsize": 20,           
    "font.size": 20,
    "legend.fontsize": 20,  
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    })

pde = "Poisson"
device = torch.device("cpu")
optimizer = "Adam"
gamma_1_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hidden_units_1 = 100
hidden_units_2 = 100
epochs=20000

mse_error = np.zeros((len(gamma_2_list), len(gamma_1_list)), dtype=object)
rel_l2_error = np.zeros_like(mse_error)
test_data = torch.linspace(-1, 1, 30).reshape(1, -1).T
true_sol = torch.sin(np.pi*test_data).detach().numpy()
for gamma_1 in gamma_1_list:
    plt.figure(figsize=(20, 10))
    label_list=[]
    for gamma_2 in gamma_2_list:
        net = PoissonNet(MLP2(num_input=1,
                              num_output=1,
                              hidden_units_1=hidden_units_1,
                              hidden_units_2 = hidden_units_2,
                              gamma_1 = gamma_1,
                              gamma_2 = gamma_2
                         ), device=device
              )
        path = os.getcwd()+ f"/results/{pde}/2layer/{optimizer}/loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}_model.pth"
        net.model.load_state_dict(torch.load(path,map_location='cpu'))
        net.model.eval()
        with torch.no_grad():
            pred = net.model(test_data)
        pred = pred.detach().numpy()
        mse_error[gamma_2_list.index(gamma_2), gamma_1_list.index(gamma_1)] = mse_vec_error(pred.ravel(), true_sol.ravel())
        rel_l2_error[gamma_2_list.index(gamma_2), gamma_1_list.index(gamma_1)] = l2_relative_loss(pred, true_sol)
        plt.plot(test_data, pred)
        label_list.append(f"$\gamma_2 = {{{gamma_2}}}$")
    plt.plot(test_data, true_sol)
    label_list.append("True solution")
    plt.xlabel("x")
    plt.ylabel("$\hat{u}(x)$", rotation=0)
    plt.grid()
    plt.legend(label_list, loc='lower center', bbox_to_anchor = [0.5, -0.2], ncols = len(label_list))
    plt.title(f"Prediction for $\gamma_1 = {{{gamma_1}}}$")
    file_name = f"plot_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}"
    fig_dir = "/content/thesis/figures_test/Poisson_test/"
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_dir + file_name + ".jpg", bbox_inches="tight", dpi=300)

"""
err_dir= f"/content/thesis/Error_tables/{pde}/"
if not os.path.isdir(err_dir):
    os.makedirs(err_dir)
pd.DataFrame(mse_error, index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list], columns=[f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]).to_csv(err_dir + f"{optimizer}_mse_table_epochs_{epochs}.csv")
pd.DataFrame(rel_l2_error, index = [f"gamma_2 = {gamma_1}" for gamma_2 in gamma_2_list], columns=[f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]).to_csv(err_dir + f"{optimizer}_rel_l2_table_epochs_{epochs}.csv")
"""