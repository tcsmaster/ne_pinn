import pandas as pd
from torch.optim import Adam
from pdes import *
from utils import *

class PoissonNet():
    """
    This class is a blueprint for a Physics-Informed Neural Network intended to
    solve the Poisson equation
                -u_xx = pi^2 sin(pi*x) -1 < x < 1
                u(-1) = u(1) = 0
    The solution is sin(pi*x)
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
        """
        Initializes the device, and moves the MLP2 to the device.
        """
        self.device=device
        self.model=model.to(self.device)
 
    def training(
        self,
        X_int_train,
        X_bc_train,
        y_bc_train,
        X_test,
        y_test,
        epochs,
        optimizer
    ):
        """
        Trains a neural network with a given optimizer for the given epochs,
        returns a pandas DataFrame with the PINN training loss, mean squared error
        over the test dataset and the relative L²-error over the test dataset.

        Parameters:
        -----------
            X_int_train: torch.tensor
                Tensor that contains the PDE residual points
            X_bc_train: torch.tensor
                Tensor containing the boundary training data points
            y_bc_train: torch.tensor
                Tensor containing the lables for X_bc_train
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
            #clear the gradients
            optimizer.zero_grad()
            #compute the forward pass on the PDE-residual training data
            u = self.model(X_int_train)
            # Calculate the PDE-residual
            loss_pde = PoissonPDE(X_int_train,u,self.device)
            # Calculate the forward pass for the boundary data
            bc_pred = self.model(X_bc_train)
            # Calculate the boundary loss
            loss_bc = MSELoss()(bc_pred, y_bc_train)
            loss = 0.5*(loss_pde + loss_bc)
            loss.backward()
            optimizer.step()
            res.loc[e, "Training Loss"] = loss.item()
            # Evaluate the model on the test set
            self.model.eval()
            with torch.no_grad():
                pred=self.model(X_test)
                mse_loss = MSELoss()(pred, y_test)
                rell2_loss = torch.linalg.vector_norm(pred- y_test) / torch.linalg.vector_norm(y_test)
                res.loc[e,"Test mse loss"]=mse_loss.item()
                res.loc[e,"Test_rel_l2_loss"]=rell2_loss.item()
        return res


def main(
    gamma_1_list,
    gamma_2_list,
    hidden_units_1,
    hidden_units_2,
    epochs,
    pde="Poisson"
):
    """
    Trains a neural network model on predicting the solution of a Poisson-equation
    and saves the resulting training and test metrics and model parameters in the
    following folders: curr_dir/results/pde/width_hidden_units_1/. If not already
    existing, the folders will be created for the results.
    
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
    
    Keyword arguments:
    pde: str
        Name of the pde we're trying to solve. Used for file and folder naming.
        Defaults to "Poisson"
    """ 
    print(f"PDE:{pde}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    X_int_train = torch.arange(-0.9,1.,0.1,device=device,requires_grad=True).reshape(1, -1).T
    bc1 = torch.tensor([-1.],device=device)
    bc2 = torch.tensor([1.], device=device)
    X_bc_train = torch.cat([bc1, bc2]).unsqueeze(1)

    y_bc1 = torch.zeros(len(bc1), device=device)
    y_bc2 = torch.zeros(len(bc2), device=device)
    y_bc_train = torch.cat([y_bc1, y_bc2]).unsqueeze(1)

    X_test = torch.linspace(-1, 1, 30, device=device).reshape(1, -1).T
    y_test = torch.sin(pi*X_test)

    file_folder = f'results/{pde}/width_{hidden_units_1}_result/'
    results_directory = os.path.join(os.getcwd(), file_folder)
    # arrays for the final metrics
    mse_error_table = np.zeros((len(gamma_2_list),len(gamma_1_list)),dtype=object)
    rel_l2_error_table = np.zeros_like(mse_error_table)

    for gamma_1 in gamma_1_list:
        for gamma_2 in gamma_2_list:
            print(
                f'''Parameters: g_1={gamma_1}, g_2={gamma_2},
                h_1={hidden_units_1}, h_2={hidden_units_2}, epochs={epochs}'''
            )
            net = PoissonNet(
                MLP2(
                    num_input=1,
                    num_output=1,
                    hidden_units_1=hidden_units_1,
                    hidden_units_2=hidden_units_2,
                    gamma_1=gamma_1,
                    gamma_2=gamma_2
                ),
                device=device
            )
            optimizer = Adam(net.model.parameters(), amsgrad=True)
            # train the network
            results = net.training(
                X_int_train=X_int_train,
                X_bc_train=X_bc_train,
                y_bc_train=y_bc_train,
                X_test=X_test,
                y_test=y_test,
                epochs=epochs,
                optimizer=optimizer
            )
            # save the final metrics to the arrays
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
            # save the csv files
            save_results(
                content=results,
                directory=results_directory,
                file_name=file_name
            )
            # save the models to the same directory
            path = os.path.join(results_directory, file_name) + '_model.pth'
            torch.save(net.model.state_dict(), path)
    # save the numpy arrays of final metrics to a DataFrame
    pd.DataFrame(
        mse_error_table,
        index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list],
        columns = [f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]
    ).to_csv(file_folder + 'mse_table.csv')
    pd.DataFrame(
        rel_l2_error_table,
        index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list],
        columns = [f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]
    ).to_csv(file_folder + f'rel_l2_table.csv')
    return

if __name__ == '__main__':
    gamma_1_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hidden_units_1=1000
    hidden_units_2=1000
    epochs=20000  
    main(
        gamma_1_list=gamma_1_list,
        gamma_2_list=gamma_2_list,
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        epochs=epochs
    )
