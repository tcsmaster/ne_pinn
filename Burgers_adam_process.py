import os
import pandas as pd
from torch.optim import Adam
from pdes import *
from utils import *

class BurgersNet():
    def __init__(self, model, device):
        self.device = device
        self.model=model.to(self.device)
    
    def training(
        self,
        X_int_train,
        X_bc_train,
        y_bc_train,
        X_ic_train,
        y_ic_train,
        X_test,
        y_test,
        epochs,
        optimizer
    ):
        res = pd.DataFrame(
            None,
            columns = ["Training Loss", "Test mse loss", "Test_rel_l2_loss"],
            dtype=float
        )
        for e in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            u = self.model(X_int_train)
            loss_pde = BurgersPDE(
                X_int_train,
                u,
                self.device
            )
            bc_pred = self.model(X_bc_train)
            loss_bc = MSELoss()(bc_pred, y_bc_train)
            ic_pred = self.model(X_ic_train)
            loss_ic = MSELoss()(ic_pred, y_ic_train)
            loss = loss_pde + loss_bc + loss_ic
            loss.backward()
            optimizer.step()
            res.loc[e, "Training Loss"] = loss.item()
            self.model.eval()
            with torch.no_grad():
                pred = self.model(X_test)
                pred = pred.cpu().detach().numpy()
                rmse_loss = mse_vec_error(pred, y_test)
                rell2_loss = l2_relative_loss(pred, y_test)
                res.loc[e, "Test mse loss"] = rmse_loss
                res.loc[e, "Test_rel_l2_loss"] = rell2_loss
        return res


def main(
    pde:str,
    gamma_1_list:list,
    gamma_2_list:list,
    hidden_units_1:int,
    hidden_units_2:int,
    epochs:int,
    directory:str,
    mse_table,
    rel_l2_table
):
    """
    Trains a neural network model on a dataset and saves the resulting 
    model accuracy and model parameters to files
    
    Parameters
    ----------
    pde: str
        Name of the pde we're trying to solve
    gamma_1: float
        the mean-field scaling parameter for the first layer
    gamma_2: float
        the mean-field scaling parameter for the second layer
    gamma_3: float
        the mean-field scaling parameter for the third layer
    hidden_units_1: int
        the number of nodes in the first layer
    hidden_units_2: int
        the number of nodes in the second layer
    hidden_units_3: int
        the number of nodes in the third layer
    directory: str
        the local where accuracy results and model parameters are saved
        (requires folders 'results' and 'models')
    """ 
    print(f"PDE:Burgers")
    print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, h_1={hidden_units_1}, h_2={hidden_units_2}, epochs={epochs}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

    print(f"Model: {net.model}")
    h = 0.05
    x = torch.arange(-1, 1 + h, h, device=device)
    t = torch.arange(0, 1 + h, h, device=device)
    X_int_train = torch.stack(
        torch.meshgrid(
            x[1:-1],
            t[1:-1],
            indexing='ij'
        )
    ).reshape(2, -1).T
    X_int_train.requires_grad=True

    bc1 = torch.stack(
        torch.meshgrid(
            x[0],
            t,
            indexing='ij'
        )
    ).reshape(2, -1).T
    bc2 = torch.stack(
        torch.meshgrid(
            x[-1],
            t,
            indexing='ij'
        )
    ).reshape(2, -1).T
    X_bc_train = torch.cat([bc1, bc2])
    y_bc1 = torch.zeros(len(bc1), device=device)
    y_bc2 = torch.zeros(len(bc2), device=device)
    y_bc_train = torch.cat([y_bc1, y_bc2]).unsqueeze(1)
    X_ic_train = torch.stack(
        torch.meshgrid(
            x,
            t[0],
            indexing='ij'
        )
    ).reshape(2, -1).T      
    y_ic_train = -torch.sin(np.pi * X_ic_train[:, 0]).unsqueeze(1)

    data = np.load("Burgers.npz")
    t, x, usol = data["t"], data["x"], data["usol"]
    X_test = torch.stack(
        torch.meshgrid(
            torch.tensor(
                [x],
                dtype=torch.float32,
                device=device
            ).squeeze(),
            torch.tensor(
                [t],
                dtype=torch.float32,
                device=device
            ).squeeze(),
            indexing="ij"
        )
    ).reshape(2, -1).T
    y_test = usol.reshape(-1, 1)
    optimizer = Adam(net.model.parameters(), amsgrad=True)
    for gamma_1 in gamma_1_list:
        for gamma_2 in gamma_2_list:
            results = net.training(
                X_int_train = X_int_train,
                X_bc_train=X_bc_train,
                y_bc_train=y_bc_train,
                X_ic_train=X_ic_train,
                y_ic_train=y_ic_train,
                X_test=X_test,
                y_test=y_test,
                epochs=epochs,
                optimizer=optimizer
            )
            mse_table[gamma_2_list.index(gamma_2), gamma_1_list.index(gamma_1)] = results["Test mse loss"]
            rel_l2_table[gamma_2_list.index(gamma_2), gamma_1_list.index(gamma_1)] = results["Test_rel_l2_loss"]
            file_name = generate_file_name(
                pde=pde,
                epochs=epochs,
                hidden_units_1=hidden_units_1,
                hidden_units_2=hidden_units_2,
                gamma_1=gamma_1,
                gamma_2=gamma_2
            )
            place = f'results/{pde}/2layer/{optimizer.__class__.__name__}/'
            results_directory = os.path.join(directory, place)
            save_results(
                results=results,
                directory=results_directory,
                file_name=file_name
            )
            path = results_directory + file_name + "_model.pth"
            torch.save(net.model.state_dict(), path)
    err_dir = f"/content/thesis/error_tables/{pde}/2layer"
    if not os.path.isdir(err_dir):
        os.makedirs(err_dir)
    pd.DataFrame(
        mse_table,
        index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list],
        columns = [f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]
    ).to_csv(err_dir + f"/{pde}_2layer_mse_table.csv")
    pd.DataFrame(
        rel_l2_error,
        index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list],
        columns = [f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]
    ).to_csv(err_dir + "/Poisson_2layer_rel_l2_table.csv")
    return

if __name__ == '__main__':
    pde='Burgers'
    gamma_1_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hidden_units_1=100
    hidden_units_2=100
    epochs = 20000
    directory=os.getcwd()
    mse_error_table = np.zeros((len(gamma_2_list), len(gamma_1_list)), dtype=object)
    rel_l2_error_table = np.zeros_like(mse_error_table)
    main(
        pde=pde,
        gamma_1_list=gamma_1_list,
        gamma_2_list=gamma_2_list,
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        epochs=epochs,
        directory=directory,
        mse_table = mse_error_table,
        rel_l2_table=rel_l2_error_table
    )
