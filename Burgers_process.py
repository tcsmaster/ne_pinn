import os
import pandas as pd
from torch.optim import Adam
from pdes import *
from utils import *

class BurgersNet():
    """
    An instance of a Physics-Informed Neural Network to solve the 
    Burgers-equation with the following boundary and initial conditions:
        u_t + u*u_x = 0.01/pi*u_xx   -1 < x < 1  ,  0 < t < 1
            u(x, 0) = -sin(pi*x)      -1 < x < 1
            u(t,-1) = u(t, 1) = 0     0 < t < 1

    Attributes:
    -----------

    model:
        A 2-layer feedforward neural network, instance of the MLP2 class.
    device:
        The computing device, where the training of the PINN happens.
        GPU if available, CPU otherwise.

    Methods:
    --------
    training:
        Trains the model with the given data and optimizer
    """
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
        """
        Trains a neural network with a given optimizer for the given epochs,
        returns a pandas DataFrame with the PINN training loss, mean squared error
        and relative L²-error over the test dataset.

        Parameters:
        -----------

            X_int_train: torch.tensor
                Tensor that contains the PDE residual points
            X_bc_train: torch.tensor
                Tensor containing the boundary training data points
            y_bc_train: torch.tensor
                Tensor containing the lables for X_bc_train
            X_ic_train: torch.tensor
                Tensor containing initial condition training data
            y_ic_train: torch.tensor
                Tensor containing the labels for X_ic_train
            X_test:torch.tensor
                Tensor containing the test data
            y_test:torch.tensor
                tensor containing the labels for X_test
            epochs:int
                the number of epochs the network is going to train
            optimizer:pytorch.optim.Optimizer
                Optimizer used for updating the weights of the network.
        """
        res = pd.DataFrame(
            None,
            columns = ["Training Loss", "Test mse loss", "Test_rel_l2_loss"],
            dtype=float
        )
        for e in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            u = self.model(X_int_train)
            loss_pde = BurgersPDE(X_int_train, u, self.device)
            bc_pred = self.model(X_bc_train)
            loss_bc = MSELoss()(bc_pred, y_bc_train)
            ic_pred = self.model(X_ic_train)
            loss_ic = MSELoss()(ic_pred, y_ic_train)
            loss = 0.5*(loss_pde + loss_bc + loss_ic)
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
    gamma_1_list,
    gamma_2_list,
    hidden_units_1,
    hidden_units_2,
    epochs,
    directory,
    mse_error_table,
    rel_l2_error_table,
    pde="Burgers"
):
    """
    Trains a neural network model on a dataset and saves the resulting test
    errors, test metrics and model parameters to files. Returns None
    
    Parameters
    ----------
        gamma_1_list: list
            the mean-field scaling parameter for the first layer
        gamma_2_list: list  
            the mean-field scaling parameter for the second layer
        hidden_units_1: int
            the number of nodes in the first layer
        hidden_units_2: int
            the number of nodes in the second layer
        epochs: int
            The number of epochs the networks are training for
        directory: str
            The directory where the script runs. If not already existing, additional
            folders will be created for the results and model parameters in the
            following fashion:
            model weights and test errors for individual networks: directory/results/pde/optimizer/
            final test metrics from all the networks: directory/error_tables/pde/optimizer/
        mse_error_table: np.ndarray
            Array that holds the final mean squared error test loss for the networks.
            Each column corresponds to a fixed gamma_1, while each row corresponds
            to a fixed gamma_2.
        rel_l2_error_table: np.ndarray
            Array that holds the final relative L^2 error test loss for the networks.
            Each column corresponds to a fixed gamma_1, while each row corresponds
            to a fixed gamma_2.
    
        Keyword arguments:
        pde: str
            The pde that is to be solved. Used for file and folder naming.
            Defaults to "Burgers"
    """ 
    print(f"PDE:{pde}")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device= torch.device('cpu')
    h = 0.05
    x = torch.arange(-1, 1 + h, h, device=device)
    t = torch.arange(0, 1 + h, h, device=device)
    # only include the inner points for the residual training data
    X_int_train = torch.stack(torch.meshgrid(
        x[1:-1],
        t[1:-1],
        indexing='ij'
    )
    ).reshape(2, -1).T
    X_int_train.requires_grad=True

    bc1 = torch.stack(torch.meshgrid(x[0],t,indexing='ij')).reshape(2, -1).T
    bc2 = torch.stack(torch.meshgrid(x[-1],t,indexing='ij')).reshape(2, -1).T
    X_bc_train = torch.cat([bc1, bc2])

    y_bc1 = torch.zeros(len(bc1), device=device)
    y_bc2 = torch.zeros(len(bc2), device=device)
    y_bc_train = torch.cat([y_bc1, y_bc2]).unsqueeze(1)

    X_ic_train = torch.stack(
        torch.meshgrid(x,t[0],indexing='ij')
    ).reshape(2, -1).T      
    y_ic_train = -torch.sin(np.pi * X_ic_train[:, 0]).unsqueeze(1)

    # testing data obtained from simulation
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

    for gamma_1 in gamma_1_list:
        for gamma_2 in gamma_2_list:
            print(
                f'''Parameters: g_1={gamma_1}, g_2={gamma_2},
                h_1={hidden_units_1}, h_2={hidden_units_2}, epochs={epochs}'''
            )
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
            optimizer = Adam(net.model.parameters(), amsgrad=True)
            results = net.training(
                X_int_train=X_int_train,
                X_bc_train=X_bc_train,
                y_bc_train=y_bc_train,
                X_ic_train=X_ic_train,
                y_ic_train=y_ic_train,
                X_test=X_test,
                y_test=y_test,
                epochs=epochs,
                optimizer=optimizer
            )
            # save the final metrics into the arrays
            mse_error_table[
                gamma_2_list.index(gamma_2),
                gamma_1_list.index(gamma_1)
            ] = results["Test mse loss"].iloc[-1]
            rel_l2_error_table[
                gamma_2_list.index(gamma_2),
                gamma_1_list.index(gamma_1)
            ] = results["Test_rel_l2_loss"].iloc[-1]
            file_name = generate_file_name(
                pde=pde,
                epochs=epochs,
                hidden_units_1=hidden_units_1,
                hidden_units_2=hidden_units_2,
                gamma_1=gamma_1,
                gamma_2=gamma_2
            )
            # save the dataframes and the model parameters
            place = f'results/{pde}/{optimizer.__class__.__name__}/'
            results_directory = os.path.join(directory, place)
            save_results(
                content=results,
                directory=results_directory,
                file_name=file_name
            )
            path = results_directory + file_name + "_model.pth"
            torch.save(net.model.state_dict(), path)
    # save the final metrics as DataFrames
    used_optimizer = optimizer.__class__.__name__
    err_dir = f"/content/thesis/Error_tables/{pde}/"
    if not os.path.isdir(err_dir):
        os.makedirs(err_dir)
    pd.DataFrame(
        mse_error_table,
        index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list],
        columns = [f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]
    ).to_csv(err_dir + f"{used_optimizer}_mse_table_epochs_{epochs}.csv")
    pd.DataFrame(
        rel_l2_error_table,
        index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list],
        columns = [f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]
    ).to_csv(err_dir + f"{used_optimizer}_rel_l2_table_epochs_{epochs}.csv")
    return

if __name__ == '__main__':
    gamma_1_list=[1.0]
    gamma_2_list=[0.7, 0.8, 0.9, 1.0]
    hidden_units_1=100
    hidden_units_2=100
    epochs = 40000
    directory=os.getcwd()
    mse_error_table = np.zeros(
        (
            len(gamma_2_list),
            len(gamma_1_list)
        ),
        dtype=object
    )
    rel_l2_error_table = np.zeros_like(mse_error_table)
    main(
        gamma_1_list=gamma_1_list,
        gamma_2_list=gamma_2_list,
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        epochs=epochs,
        directory=directory,
        mse_error_table = mse_error_table,
        rel_l2_error_table=rel_l2_error_table
    )
