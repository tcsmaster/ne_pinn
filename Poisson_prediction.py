import torch
import pandas as pd
import matplotlib.pyplot as plt
from Poisson_process import PoissonNet
from utils import *

"""
This script tries to predict the solution of the Poisson equation
             -u_xx = pi^2 sin(pi*x) -1 < x < 1
              u(-1) = u(1) = 0

over the test data for different gamma_1 and gamma_2 combinations,plots the
results and the true solution, and saves the plots in the folder
'curr_dir/prediction_figures/Poisson'
"""
plt.rcParams.update({                   # matplotlib parameter settings
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
hidden_units_1 = 1000
hidden_units_2 = 1000
epochs=20000

# Test data to predict on
test_data = torch.linspace(-1, 1, 30).reshape(1, -1).T
true_sol = torch.sin(np.pi*test_data).numpy()
for gamma_1 in gamma_1_list:
    plt.figure(figsize=(20, 10))
    # list for plot labels
    label_list=[]
    for gamma_2 in gamma_2_list:
        # define the model architecture
        net = PoissonNet(
            MLP2(
                num_input=1,
                num_output=1,
                hidden_units_1=hidden_units_1,
                hidden_units_2 = hidden_units_2,
                gamma_1 = gamma_1,
                gamma_2 = gamma_2
            ), device=device
        )
        #load the saved model weights
        path = os.getcwd()+ f"/results/{pde}/width_{hidden_units_1}/loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}_model.pth"
        net.model.load_state_dict(torch.load(path,map_location='cpu'))
        net.model.eval()
        with torch.no_grad():
            pred = net.model(test_data)
        pred = pred.numpy()
        plt.plot(test_data, pred)
        label_list.append(f"$\gamma_2 = {{{gamma_2}}}$")
    plt.plot(test_data, true_sol)
    label_list.append("True solution")
    plt.xlabel("x")
    plt.ylabel("$\hat{u}(x)$", rotation=0)
    plt.grid()
    plt.legend(label_list, loc='lower center', bbox_to_anchor = [0.5, -0.2], ncols = len(label_list))
    plt.title(f"Prediction for $\gamma_1 = {{{gamma_1}}}$ for $N = {{{hidden_units_1}}}$")
    # create filename and save the plot
    file_name = f"plot_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}.jpg"
    fig_dir = os.getcwd() + f"/prediction_figures/{pde}/width_{hidden_units_1}_prediction_plot/"
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_dir + file_name, bbox_inches="tight", dpi=300)
