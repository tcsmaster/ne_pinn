import torch
import pandas as pd
from Burgers_process import *
from utils import *


pde = "Burgers"
epochs=20000
hidden_units_1 = 100
hidden_units_2 = 100

gamma_1_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

mse_error = np.zeros((len(gamma_2_list), len(gamma_1_list)), dtype=object)
rel_l2_error = np.zeros_like(mse_error)
data = np.load("Burgers.npz")
t, x, usol = data["t"], data["x"], data["usol"]
test_data = torch.stack(torch.meshgrid(torch.tensor([x], dtype=torch.float32).squeeze(),
                                           torch.tensor([t], dtype=torch.float32).squeeze(),
                                           indexing="ij")).reshape(2, -1).T
true_sol = usol.reshape(-1, 1)
for gamma_1 in gamma_1_list:
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
        path = os.getcwd()+ f"/results/{pde}/2layer/Adam/loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}_model.pth"
        net.model.load_state_dict(torch.load(path,map_location='cpu'))
        net.model.eval()
        with torch.no_grad():
                pred = net.model(test_data)
        pred = pred.detach().numpy()
        mse_error[gamma_2_list.index(gamma_2), gamma_1_list.index(gamma_1)] = mse_vec_error(pred, true_sol)
        rel_l2_error[gamma_2_list.index(gamma_2), gamma_1_list.index(gamma_1)] = l2_relative_loss(pred, true_sol)

err_dir = f"/content/thesis/Error_tables/{pde}/2layer"
if not os.path.isdir(err_dir):
    os.makedirs(err_dir)
pd.DataFrame(mse_error, index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list], columns = [f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]).to_csv(err_dir + "/Burgers_2layer_mse_table.csv")
pd.DataFrame(rel_l2_error, index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list], columns = [f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]).to_csv(err_dir + "/Burgers_2layer_rel_l2_table.csv")