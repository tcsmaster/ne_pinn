import os
from numpy import pi
import pandas as pd
from torch.optim import LBFGS, Adam
from models import *
from pdes import PoissonPDE
from utils import *

class PoissonNet():
    """
    This class is a blueprint for solving a 1D Heat equation with Dirichlet BC
    """
    def __init__(self, model, device):
        self.device=device
        self.model = model.to(self.device)
 
    
    def training(self,
                 X_int_train,
                 X_bc_train,
                 y_bc_train,
                 epochs
        ):
        res = pd.DataFrame(None, columns=['Training Loss', 'Test Loss'], dtype=float)
        optimizer = Adam(self.model.parameters())
        for e in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            u = self.model(X_int_train)
            loss_pde = PoissonPDE(X_int_train, u, self.device)
            y_bc_pred = self.model(X_bc_train)
            loss_bc = torch.nn.MSELoss()(y_bc_pred, y_bc_train)               
            loss = loss_pde + loss_bc
            res.loc[e, 'Training Loss'] = loss.item()
            loss.backward()
            optimizer.step()
        return res



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
                              gamma_2=gamma_2,
                              sampler=sampler
                         ),
                         device=device
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
    print(f"Model: {net.model}")
    
    X_int_train = torch.arange(-0.9, 1., 0.1, device=device).reshape(1, -1).T
    X_int_train.requires_grad=True

    bc1 = torch.Tensor([-1.], device=device)
    bc2 = torch.Tensor([1.], device=device)
    X_bc_train = torch.cat([bc1, bc2]).unsqueeze(1)

    y_bc1 = torch.zeros(len(bc1))
    y_bc2 = torch.zeros(len(bc2))
    y_bc_train = torch.cat([y_bc1, y_bc2]).unsqueeze(1)
    results = net.training(X_int_train = X_int_train,
                           X_bc_train=X_bc_train,
                           y_bc_train = y_bc_train,
                           epochs = adam_epochs
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
    gamma_1_list = [0.5,0.7,1.0]
    gamma_2_list = [0.5,0.7,1.0]
    gamma_3_list = [0.5,0.7,1.0]
    hidden_units_1=100
    hidden_units_2=100
    hidden_units_3=100
    adam_epochs=20000
    directory=os.getcwd()
    #sampler_list = ['random', 'Halton', 'LHS', 'Sobol']
    for gamma_1 in gamma_1_list:
        for gamma_2 in gamma_2_list:            
            main(pde=pde,
             gamma_1=gamma_1,
             gamma_2=gamma_2,
             hidden_units_1=hidden_units_1,
             hidden_units_2=hidden_units_2,
             epochs=adam_epochs,
             directory=directory
        )