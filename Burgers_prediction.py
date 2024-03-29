import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Burgers_process import *
from utils import *

"""
This script creates predictions of the PINNs trained on the Burgers equation

        u_t + u*u_x = 0.01/pi*u_xx   -1 < x < 1  ,  0 < t < 1
            u(x, 0) = -sin(pi*x)      -1 < x < 1
            u(t,-1) = u(t, 1) = 0     0 < t < 1

at t=0 (initial condition), t=0.25, t=0.5 and t=0.75 for different gamma_1 and gamma_2
combinations, groups the plots according to gamma_1 and saves the plots to the
directory 'curr_dir/prediction_plots/Burgers'. 
"""
plt.rcParams.update({                   # matplotlib parameter settings
    "font.monospace": [],
    "figure.figsize": (20,12),
    "axes.labelsize": 20,           
    "font.size": 20,
    "legend.fontsize": 20,  
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

pde = "Burgers"
epochs=40000
hidden_units_1 = 100
hidden_units_2 = 100
optimizer="Adam"
gamma_1_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
t_space = [0, 0.25, 0.5, 0.75]

# load the testing data
data = np.load("Burgers.npz")
t, x, usol = data["t"], data["x"], data["usol"]

for gamma_1 in gamma_1_list:
    fig, axs = plt.subplots(ncols=2, nrows=2)
    for ax, time in zip(axs.ravel(), t_space):
        #create pytorch tensors from the numpy arrays at the given timepoint
        test_data = torch.stack(
            torch.meshgrid(
                torch.tensor([x],dtype=torch.float32).squeeze(),
                torch.tensor([time],dtype=torch.float32),
                indexing="ij"
            )
        ).reshape(2, -1).T
        # get the corresponding solution
        true_sol = usol[:, np.where(t == time)[0]]
        for gamma_2 in gamma_2_list:
            # define the model architecture
            net = BurgersNet(
                MLP2(
                    num_input=2,
                    num_output=1,
                    hidden_units_1=hidden_units_1,
                    hidden_units_2=hidden_units_2,
                    gamma_1=gamma_1,
                    gamma_2=gamma_2
                ),
                device=torch.device('cpu')
            )
            #load the saved model weights
            path = os.getcwd()+ f"/results/{pde}/width_{hidden_units_1}/loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}_model.pth"
            net.model.load_state_dict(torch.load(path,map_location='cpu'))
            net.model.eval()
            with torch.no_grad():
                pred = net.model(test_data)
            pred = pred.numpy()
            ax.plot(x, pred, label = f"$\gamma_2 = {{{gamma_2}}}$")
        ax.plot(x, true_sol, label="True solution")
        ax.set_xlabel("x")
        ax.set_ylabel("$\hat{u}(x)$", rotation=0)
        ax.grid()
        ax.set_title(f"t={time}")
    plt.suptitle(f"Prediction for $\gamma_1 = {{{gamma_1}}}$ for $N={{{hidden_units_1}}}$")
    fig.legend(
        [f"$\gamma_2 = {{{gamma_2}}}$" for gamma_2 in gamma_2_list] + ["True solution"],
        loc="lower center",
        bbox_to_anchor = [0.53, -0.05],
        ncols = len(gamma_2_list)+1
    )
    fig.tight_layout()
    #save the figure
    file_name = f"plot_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}"
    fig_dir = os.getcwd() + f"/prediction_figures/{pde}/width_{hidden_units_1}/"
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_dir + file_name + ".jpg",  bbox_inches="tight", dpi=300)


