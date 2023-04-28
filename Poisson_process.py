import pandas as pd
from torch.optim import Adam
from pdes import *
from utils import *

class PoissonNet():
    """
    This class is a blueprint for a Physics-Informed Neural Network intended to
    solve a 1D Poisson equation with Dirichlet BC and forcing term pi^2*sin(pi*x).
    The solution is sin(pi*x)
    """
    def __init__(self, model, device):
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
        res = pd.DataFrame(
            None,
            columns = ["Training Loss", "Test mse loss", "Test_rel_l2_loss"],
            dtype=float
        )
        for e in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            u = self.model(X_int_train)
            loss_pde = PoissonPDE(
                X_int_train,
                u,
                self.device
            )
            bc_pred = self.model(X_bc_train)
            loss_bc = MSELoss()(bc_pred, y_bc_train)
            loss = 0.5*(loss_pde + loss_bc)
            loss.backward()
            optimizer.step()
            res.loc[e, "Training Loss"] = loss.item()
            self.model.eval()
            with torch.no_grad():
                pred=self.model(X_test)
                mse_loss=mse_vec_error(pred, y_test)
                rell2_loss=l2_relative_loss(pred, y_test)
                res.loc[e,"Test mse loss"]=mse_loss
                res.loc[e,"Test_rel_l2_loss"]=rell2_loss
        return res


def main(
    pde:str,
    gamma_1_list:list,
    gamma_2_list:list,
    hidden_units_1:int,
    hidden_units_2:int,
    epochs:int,
    directory:str,
    mse_error_table,
    rel_l2_error_table
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
    #TODO: update the docstring with the relevant parameters
    print(f"PDE:{pde}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    X_int_train = torch.arange(
        -0.9,
        1.,
        0.1,
        device=device,
        requires_grad=True
    ).reshape(1, -1).T
    bc1 = torch.tensor([-1.],device=device)
    bc2 = torch.tensor([1.], device=device)
    X_bc_train = torch.cat([bc1, bc2]).unsqueeze(1)

    y_bc1 = torch.zeros(len(bc1), device=device)
    y_bc2 = torch.zeros(len(bc2), device=device)
    y_bc_train = torch.cat([y_bc1, y_bc2]).unsqueeze(1)

    X_test = torch.linspace(-1, 1, 30, device=device).reshape(1, -1).T
    y_test = torch.sin(pi*X_test)
    for gamma_1 in gamma_1_list:
        for gamma_2 in gamma_2_list:
            print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, h_1={hidden_units_1}, h_2={hidden_units_2}, epochs={epochs}")
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
            print(f"Model: {net.model}")
            optimizer = Adam(net.model.parameters(), amsgrad=True)
            results = net.training(
                X_int_train=X_int_train,
                X_bc_train=X_bc_train,
                y_bc_train=y_bc_train,
                X_test=X_test,
                y_test=y_test,
                epochs=epochs,
                optimizer=optimizer
            )
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
            results_directory = os.path.join(
                directory,
                f'results/{pde}/2layer/{optimizer.__class__.__name__}/'
            )
            save_results(
                results=results,
                directory=results_directory,
                file_name=file_name
            )
            path = os.path.join(results_directory, file_name) + '_model.pth'
            torch.save(net.model.state_dict(), path)
    err_dir= f"/content/thesis/Error_tables/{pde}/"
    if not os.path.isdir(err_dir):
        os.makedirs(err_dir)
    pd.DataFrame(
        mse_error_table,
        index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list],
        columns=[f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]
    ).to_csv(err_dir + f"{optimizer.__class__.__name__}_mse_table_epochs_{epochs}.csv")
    pd.DataFrame(
        rel_l2_error_table,
        index = [f"gamma_2 = {gamma_2}" for gamma_2 in gamma_2_list],
        columns=[f"gamma_1 = {gamma_1}" for gamma_1 in gamma_1_list]
    ).to_csv(err_dir + f"{optimizer.__class__.__name__}_rel_l2_table_epochs_{epochs}.csv")     
    return

if __name__ == '__main__':
    pde='Poisson'
    gamma_1_list = [0.5, 0.6, 0.7,0.8, 0.9, 1.0]
    gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hidden_units_1=100
    hidden_units_2=100
    epochs=20000
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
        mse_error_table=mse_error_table,
        rel_l2_error_table = rel_l2_error_table
    )
