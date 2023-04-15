import os
from numpy import pi
import pandas as pd
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from models import *
from torch.utils.data import Dataset, DataLoader
from pdes import PoissonPDE
from utils import *

class PoissonNet():
    """
    This class is a blueprint for solving a 1D Poisson equation with Dirichlet BC
    """
    def __init__(self, model, device):
        self.device=device
        self.model = model.to(self.device)
 
    
    def training(self,
                 train_data,
                 boundary_data,
                 test_points,
                 true_sol,
                 adam_epochs,
                 optimizer
        ):
        training_data = DataLoader(train_data, batch_size=1)
        bc_data = DataLoader(boundary_data, batch_size=1)
        res = pd.DataFrame(None,
                           columns = ["Training Loss", "Test_rmse_loss", "Test_rel_l2_loss"],
                           dtype=float
              )
        for e in range(adam_epochs):
            self.model.train()
            for batch in training_data:
                optimizer.zero_grad()
                u = self.model(batch)
                loss_pde = PoissonPDE(batch,
                                      u,
                                      self.device
                           )
                loss_pde.backward()
                optimizer.step()
            for x, y in bc_data:
                optimizer.zero_grad()
                pred = self.model(x)
                loss = MSELoss()(pred, y)
                loss.backward()
                optimizer.step()
            res.loc[e, "Training Loss"] = loss.item()
            self.model.eval()
            with torch.no_grad():
                pred = self.model(test_points)
                rmse_loss = rmse_vec_error(pred, true_sol)
                rell2_loss = l2_relative_loss(pred, true_sol)
                res.loc[e, "Test_rmse_loss"] = rmse_loss
                res.loc[e, "Test_rel_l2_loss"] = rell2_loss

        return res

class train_loader(Dataset):
    
    def __init__(self, X):
      self.X = X

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index, :]

class bc_loader(Dataset):
    
    def __init__(self, X, y):
      self.X = X
      self.y = y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index, :], self.y[index, :]

def main(pde,
         gamma_1,
         gamma_2,
         hidden_units_1,
         hidden_units_2,
         adam_epochs,
         directory,
         gamma_3 = None,
         hidden_units_3 = None,
         sampler=None
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
    print(f"PDE:{pde}")
    if (not gamma_3):
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, h_1={hidden_units_1}, h_2={hidden_units_2}, epochs={adam_epochs}")
    else:
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, g_3={gamma_3}, h_1={hidden_units_1}, h_2={hidden_units_2}, h_3={hidden_units_3}, epochs = {adam_epochs}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if (not gamma_3):
        net = PoissonNet(MLP2(num_input=1,
                              num_output=1,
                              hidden_units_1=hidden_units_1,
                              hidden_units_2=hidden_units_2,
                              gamma_1=gamma_1,
                              gamma_2=gamma_2
                         ),
                         device=device
              )
        learning_rate_fc1 = 1.0 / ((hidden_units_1 ** (1 - 2 * gamma_1)) * (hidden_units_2 ** (3 - 2 * gamma_2)))
        learning_rate_fc2 = 1.0 / ((hidden_units_1 ** (1 - 2 * gamma_1)) * (hidden_units_2 ** (2 - 2 * gamma_2)))
        learning_rate_fc3 = 1.0 / (hidden_units_2 ** (2 - 2 * gamma_2))
        optimizer = SGD([{"params": net.model.fc1.parameters(), "lr": learning_rate_fc1},
                         {"params": net.model.fc2.parameters(), "lr": learning_rate_fc2},
                         {"params": net.model.fc3.parameters(), "lr": learning_rate_fc3}],
                         lr = 1.0
                    )
    else:
        net = PoissonNet(MLP3(num_input=1,
                              num_output=1,
                              hidden_units_1=hidden_units_1,
                              hidden_units_2=hidden_units_2,
                              hidden_units_3=hidden_units_3,
                              gamma_1=gamma_1,
                              gamma_2=gamma_2,
                              gamma_3=gamma_3,
                              sampler=sampler
                         ),
                         device=device
              )
        learning_rate_fc1 = 1.0 / ((hidden_units_1 ** (1 - 2 * gamma_1)) * (hidden_units_2 ** (2 - 2 * gamma_2)) * (hidden_units_3**(3 - 2 * gamma_3)))
        learning_rate_fc2 = 1.0 / ((hidden_units_1 ** (1 - 2 * gamma_1)) * (hidden_units_2 ** (1 - 2 * gamma_2)) * (hidden_units_3**(3 - 2 * gamma_3)))
        learning_rate_fc3 = 1.0 / ((hidden_units_2 ** (1 - 2 * gamma_2)) * (hidden_units_3 ** (2 - 2 * gamma_3)))
        learning_rate_fc4 = 1.0 / (hidden_units_3 ** (2 - 2 * gamma_3))
        optimizer = SGD([{"params": net.model.fc1.parameters(), "lr": learning_rate_fc1},
                         {"params": net.model.fc2.parameters(), "lr": learning_rate_fc2},
                         {"params": net.model.fc3.parameters(), "lr": learning_rate_fc3},
                         {"params": net.model.fc4.parameters(), "lr": learning_rate_fc4}],
                         lr = 1.0
                    )
    print(f"Model: {net.model}")
    
    X_int_train = torch.arange(-0.9,
                               1.,
                               0.1,
                               device=device,
                               requires_grad=True
                  ).reshape(1, -1).T
    train_data = train_loader(X_int_train)
    bc1 = torch.tensor([-1.], device=device)
    bc2 = torch.tensor([1.], device=device)
    X_bc_train = torch.cat([bc1, bc2]).unsqueeze(1)

    y_bc1 = torch.zeros(len(bc1), device=device)
    y_bc2 = torch.zeros(len(bc2), device=device)
    y_bc_train = torch.cat([y_bc1, y_bc2]).unsqueeze(1)
    boundary_data = bc_loader(X_bc_train, y_bc_train)

    test_points = torch.linspace(-1, 1, 30, device=device).reshape(1, -1).T
    true_sol = torch.sin(pi*test_points)
    results = net.training(train_data = train_data,
                           boundary_data = boundary_data,
                           test_points = test_points,
                           true_sol=true_sol,
                           adam_epochs = adam_epochs,
                           optimizer=optimizer
              )



    # Save accuracy results
    if not gamma_3:
        file_name = generate_file_name(pde=pde,
                                       epochs=adam_epochs,
                                       hidden_units_1=hidden_units_1,
                                       hidden_units_2=hidden_units_2,
                                       gamma_1=gamma_1,
                                       gamma_2=gamma_2
        )
        results_directory = os.path.join(directory, f'results/{pde}/2layer/normalized/')
    else:
        file_name = generate_file_name(pde=pde,
                                   epochs=adam_epochs,
                                   hidden_units_1=hidden_units_1,
                                   hidden_units_2=hidden_units_2,
                                   gamma_1=gamma_1,
                                   gamma_2=gamma_2,
                                   hidden_units_3=hidden_units_3,
                                   gamma_3=gamma_3
        )
        results_directory = os.path.join(directory, f'results/{pde}/3layer/normalized/')
    save_results(results=results,
                 directory=results_directory,
                 file_name=file_name
    )
    path = os.path.join(results_directory, file_name) + '_model.pth'
    torch.save(net.model.state_dict(), path)
    return

if __name__ == '__main__':
    pde='Poisson'
    gamma_1_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gamma_3_list = [0.5,0.7,0.9]
    hidden_units_1=100
    hidden_units_2=100
    hidden_units_3=100
    adam_epochs=2000
    directory=os.getcwd()
    for gamma_1 in gamma_1_list:
        for gamma_2 in gamma_2_list:         
               main(pde=pde,
                gamma_1=gamma_1,
                gamma_2=gamma_2,
               hidden_units_1=hidden_units_1,
             hidden_units_2=hidden_units_2,
             adam_epochs=adam_epochs,
             directory=directory)