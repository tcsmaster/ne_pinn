import pandas as pd
from utils import *
from Burgers_process import BurgersNet
from numpy import pi
device=torch.device("cpu")
gamma_1_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hidden_units_1=500
hidden_units_2=500
epochs=40000
pde = 'Burgers'
directory=os.getcwd()

data = np.load("Burgers.npz")
t, x, usol = data["t"], data["x"], data["usol"]

X_test = torch.stack(
    torch.meshgrid(
        torch.tensor([x],dtype=torch.float32,device=device).squeeze(),
        torch.tensor([t],dtype=torch.float32,device=device).squeeze(),
        indexing="ij"
    )
).reshape(2, -1).T
y_test = usol.reshape(-1, 1)
mse_error_table = np.zeros(
        (
            len(gamma_2_list),
            len(gamma_1_list)
        ),
        dtype=object
    )
rel_l2_error_table = np.zeros_like(mse_error_table)

for gamma_1 in gamma_1_list:
    for gamma_2 in gamma_2_list:
        net = BurgersNet(
                MLP2(
                    num_input=2,
                    num_output=1,
                    hidden_units_1=hidden_units_1,
                    hidden_units_2=hidden_units_2,
                    gamma_1=gamma_1,
                    gamma_2=gamma_2
                ),
                device=device
            )
        file_name =  directory + f'/results/{pde}/Adam/loss_{pde}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_gamma1_{gamma_1}_gamma2_{gamma_2}_epochs_{epochs}_model.pth'
        net.model.load_state_dict(torch.load(file_name, map_location=torch.device("cpu")))
        net.model.eval()
        with torch.no_grad():
            pred=net.model(X_test)
            pred = pred.detach().numpy()
        mse_error_table[
                gamma_2_list.index(gamma_2),
                gamma_1_list.index(gamma_1)
                ] = mse_vec_error(pred, y_test)
        rel_l2_error_table[
                gamma_2_list.index(gamma_2),
                gamma_1_list.index(gamma_1)
                ] = l2_relative_loss(pred, y_test)
used_optimizer = 'Adam'
err_dir = f"/content/thesis/Error_tables/{pde}/"
if not os.path.isdir(err_dir):
    os.makedirs(err_dir)
pd.DataFrame(
        mse_error_table,
        index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list],
        columns = [f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]
).to_csv(err_dir + f'''{used_optimizer}_mse_table_epochs_{epochs}.csv''')
pd.DataFrame(
        rel_l2_error_table,
        index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list],
        columns = [f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]
).to_csv(err_dir + f'''{used_optimizer}_rel_l2_table_epochs_{epochs}.csv''')

