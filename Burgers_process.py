import os
import pandas as pd
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from pdes import *
from utils import *

class BurgersNet():
    def __init__(self, model, device):
        self.device = device
        self.model=model.to(self.device)
    
    def training(self,
                 train_data,
                 boundary_data,
                 init_data,
                 test_points,
                 true_sol,
                 epochs,
                 optimizer
        ):
        training_data = DataLoader(train_data, batch_size=1)
        bc_data = DataLoader(boundary_data, batch_size=1)
        ic_data = DataLoader(init_data, batch_size=1)
        res = pd.DataFrame(None,
                           columns = ["Training Loss", "Test_mse_loss", "Test_rel_l2_loss"],
                           dtype=float
              )
        for e in range(epochs):
            self.model.train()
            for batch in training_data:
                optimizer.zero_grad()
                u = self.model(batch)
                loss_pde = BurgersPDE(batch, u, self.device)
                optimizer.step()
            for x, y in bc_data:
                optimizer.zero_grad()
                pred = self.model(x)
                loss = torch.nn.MSELoss()(pred, y)
                loss.backward()
                optimizer.step()
            for x, y in ic_data:
                optimizer.zero_grad()
                pred = self.model(x)
                loss = torch.nn.MSELoss()(pred, y)
                loss.backward()
                optimizer.step()
            res.loc[e, "Training Loss"] = loss.item()
            self.model.eval()
            with torch.no_grad():
                pred = self.model(test_points)
                pred = pred.cpu().detach().numpy()
                rmse_loss = rmse_vec_error(pred, true_sol)
                rell2_loss = l2_relative_loss(pred, true_sol)
                res.loc[e, "Test_mse_loss"] = rmse_loss
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


def main(pde:str,
         gamma_1:float,
         gamma_2:float,
         hidden_units_1:int,
         hidden_units_2:int,
         epochs:int,
         directory,
         sampler=None,
         gamma_3 = None,
         hidden_units_3 = None
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
    if (not gamma_3):
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, h_1={hidden_units_1}, h_2={hidden_units_2}, epochs={epochs}")
    else:
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, g_3={gamma_3}, h_1={hidden_units_1}, h_2={hidden_units_2}, h_3={hidden_units_3}, epochs = {epochs}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if (not gamma_3):
        net = BurgersNet(MLP2(num_input=2,
                              num_output=1,
                              hidden_units_1=hidden_units_1,
                              hidden_units_2=hidden_units_2,
                              gamma_1=gamma_1,
                              gamma_2=gamma_2,
                              sampler=sampler
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
        net = BurgersNet(MLP3(num_input=2,
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
    if not sampler:
        h = 0.05
        x = torch.arange(-1, 1 + h, h, device=device)
        t = torch.arange(0, 1 + h, h, device=device)
        X_int_train = torch.stack(torch.meshgrid(x[1:-1],
                                                 t[1:-1],
                                                 indexing='ij'
                                  )
                      ).reshape(2, -1).T
        X_int_train.requires_grad=True
        train_data = train_loader(X_int_train)
        bc1 = torch.stack(torch.meshgrid(x[0],
                                         t,
                                         indexing='ij'
                          )
              ).reshape(2, -1).T
        bc2 = torch.stack(torch.meshgrid(x[-1],
                                         t,
                                         indexing='ij'
                          )
              ).reshape(2, -1).T
        X_bc_train = torch.cat([bc1, bc2])
        y_bc1 = torch.zeros(len(bc1), device=device)
        y_bc2 = torch.zeros(len(bc2), device=device)
        y_bc_train = torch.cat([y_bc1, y_bc2]).unsqueeze(1)
        boundary_data = bc_loader(X_bc_train, y_bc_train)
        X_ic_train = torch.stack(torch.meshgrid(x,
                                                t[0],
                                                indexing='ij'
                                 )
                     ).reshape(2, -1).T      
        y_ic_train = -torch.sin(np.pi * X_ic_train[:, 0]).unsqueeze(1)
        init_data = bc_loader(X_ic_train, y_ic_train)

        data = np.load("Burgers.npz")
        t, x, usol = data["t"], data["x"], data["usol"]
        test_points = torch.stack(torch.meshgrid(torch.tensor([x],
                                                              dtype=torch.float32
                                                 )
                                                 .squeeze(),
                                                 torch.tensor([t],
                                                              dtype=torch.float32
                                                 )
                                                 .squeeze(),
                                                 indexing="ij"
                                  )
                      ).reshape(2, -1).T
        true_sol = usol.reshape(-1, 1)
        results = net.training(train_data = train_data,
                               boundary_data = boundary_data,
                               init_data = init_data,
                               test_points=test_points,
                               true_sol=true_sol,
                               epochs=epochs,
                               optimizer=optimizer
                  )

        if not gamma_3:
            file_name = generate_file_name(pde=pde,
                                           epochs=epochs,
                                           hidden_units_1=hidden_units_1,
                                           hidden_units_2=hidden_units_2,
                                           gamma_1=gamma_1,
                                           gamma_2=gamma_2
                        )
            place = f'results/{pde}/2layer/{optimizer.__class__.__name__}/'
            results_directory = os.path.join(directory, place)
        else:
            file_name = generate_file_name(pde=pde,
                                           epochs=epochs,
                                           hidden_units_1=hidden_units_1,
                                           hidden_units_2=hidden_units_2,
                                           gamma_1=gamma_1,
                                           gamma_2=gamma_2,
                                           hidden_units_3=hidden_units_3,
                                           gamma_3=gamma_3
                        )
            place = f'results/{pde}/3layer/{optimizer.__class__.__name__}/'
            results_directory = os.path.join(directory, place)
        save_results(results=results,
                 directory=results_directory,
                 file_name=file_name
        )
        path = results_directory + file_name + "_model.pth"
        torch.save(net.model.state_dict(), path)
    return

if __name__ == '__main__':
    pde='Burgers'
    gamma_1_list = [0.6]
    gamma_2_list = [0.6, 0.7, 0.8, 0.9]
    gamma_3_list = [0.5, 0.7, 0.9]
    hidden_units_1=100
    hidden_units_2=100
    hidden_units_3=100
    epochs = 2000
    directory=os.getcwd()
    for gamma_1 in gamma_1_list:
        for gamma_2 in gamma_2_list:
            main(pde=pde,
                     gamma_1=gamma_1,
                     gamma_2=gamma_2,
                     hidden_units_1=hidden_units_1,
                     hidden_units_2=hidden_units_2,
                     epochs=epochs,
                     directory=directory
                )
