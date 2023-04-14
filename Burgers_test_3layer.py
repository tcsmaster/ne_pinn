import torch
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Burgers_process import *
from utils import *

pde = "Burgers"
epochs=1000
hidden_units_1 = 100
hidden_units_2 = 100
hidden_units_3 = 100
gamma_1_list = [0.5, 0.7, 0.9]
gamma_2_list = [0.5, 0.7, 0.9]
gamma_3_list = [0.5, 0.7, 0.9]
rmse_error = np.zeros((9, 12), dtype=object)
rel_l2_error = np.zeros((9,12), dtype=object)

t_space = [0, 0.25, 0.5, 0.75]
data = np.load("Burgers.npz")
t, x, usol = data["t"], data["x"], data["usol"]

for el in itertools.product(gamma_2_list, gamma_3_list):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize = (20, 10), dpi=300)
    for ax, time in zip(axs.ravel(), t_space):
        test_data = torch.stack(torch.meshgrid(torch.tensor([time], dtype=torch.float32, device=torch.device("cpu")),
                                               torch.tensor([x], dtype=torch.float32, device=torch.device("cpu")).squeeze(),
                                               indexing="ij")).reshape(2, -1).T
        true_sol = usol[:, np.where(t == time)[0]]
        for gamma_1 in gamma_1_list:
            net = BurgersNet(MLP3(num_input=2,
                                  num_output=1,
                                  hidden_units_1=hidden_units_1,
                                  hidden_units_2=hidden_units_2,
                                  hidden_units_3=hidden_units_3,
                                  gamma_1=gamma_1,
                                  gamma_2=el[0],
                                  gamma_3=el[1]
                             ),
                             device=torch.device('cpu')
                  )
            path = os.getcwd()+ f"/results/{pde}/3layer/normalized/SGD/loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_hidden3_{hidden_units_3}_gamma1_{gamma_1}_gamma2_{el[0]}_gamma3_{el[1]}_epochs_{epochs}_model.pth"
            net.model.load_state_dict(torch.load(path,map_location='cpu'))
            net.model.eval()
            with torch.no_grad():
                pred = net.model(test_data)
            pred = pred.detach().numpy()
            ax.plot(x, pred, label = f"$\gamma_1 = {{{gamma_1}}}$")
        ax.plot(x, true_sol, label="True solution")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid()
        ax.legend(loc="best")
        ax.set_title(f"t={time}")
    plt.suptitle(f"Prediction for $\gamma_2 = {{{el[0]}}}, \gamma_3 = {{{el[1]}}}$")
    directory= "/content/thesis/figures/Burgers_test/"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    file_name = f"plot_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_hidden3_{hidden_units_3}_gamma1_{gamma_1}_gamma2_{el[0]}_gamma3_{el[1]}_epochs_{epochs}"
    plt.savefig(directory + file_name + ".jpg")

