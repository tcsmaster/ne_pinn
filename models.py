import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from helpers import *
from pdes import *


class MLP2(nn.Module):
    """Multi-layer perceptron with two hidden layers
    
    Attributes
    ----------
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    gamma_1: float
        the mean-field scaling parameter for the first hidden layer
    gamma_2: float
        the mean-field scaling parameter for the second hidden layer
    """
    
    def __init__(self, num_input, num_output, hidden_units_1, hidden_units_2, gamma_1, gamma_2, sampler = None):
        super(MLP2, self).__init__()
        
        # Parameters
        self.num_input = num_input
        self.num_output = num_output
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.sampler = sampler
        # Layers
        self.fc1 = nn.Linear(self.num_input, self.hidden_units_1)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        self.fc2 = nn.Linear(self.hidden_units_1, self.hidden_units_2)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)
        self.fc3 = nn.Linear(self.hidden_units_2, self.num_output)
        nn.init.uniform_(self.fc3.weight, a=0.0, b=1.0)
    
    def forward(self, x):
        if not self.sampler:
            scaling_1 = self.hidden_units_1 ** (-self.gamma_1)
            x = scaling_1 * torch.tanh(self.fc1(x))
            scaling_2 = self.hidden_units_2**(-self.gamma_2)
            x = scaling_2 * torch.tanh(self.fc2(x))
            x = self.fc3(x)
            return x
        else:
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            return x
    
class MLP3(nn.Module):
    """Multi-layer perceptron with three hidden layers
    
    Attributes
    ----------
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    hidden_units_3: int
        the number of nodes in the third hidden layer
    gamma_1: float
        the mean-field scaling parameter for the first hidden layer
    gamma_2: float
        the mean-field scaling parameter for the second hidden layer
    gamma_3: float
        the mean-field scaling parameter for the third hidden layer
    """
    
    def __init__(self, num_input, num_output, hidden_units_1, hidden_units_2, hidden_units_3, gamma_1, gamma_2, gamma_3, sampler = None):
        super(MLP3, self).__init__()
        
        # Parameters
        self.num_input = num_input
        self.num_output = num_output
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.hidden_units_3 = hidden_units_3
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.gamma_3 = gamma_3
        self.sampler = sampler
        
        # Layers
        self.fc1 = nn.Linear(self.num_input, self.hidden_units_1)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        self.fc2 = nn.Linear(self.hidden_units_1, self.hidden_units_2)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)
        self.fc3 = nn.Linear(self.hidden_units_2, self.hidden_units_3)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1.0)
        self.fc4 = nn.Linear(self.hidden_units_3, self.num_output)
        nn.init.uniform_(self.fc4.weight, a=0.0, b=1.0)
    
    def forward(self, x):
        if not self.sampler:
            scaling_1 = self.hidden_units_1 ** (-self.gamma_1)
            x = scaling_1 * torch.tanh(self.fc1(x))
            scaling_2 = self.hidden_units_2**(-self.gamma_2)
            x = scaling_2 * torch.tanh(self.fc2(x))
            scaling_3 = self.hidden_units_3**(-self.gamma_3)
            x = scaling_3 * torch.tanh(self.fc3(x))
            x = self.fc4(x)
            return x
        
        else:
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            x = self.fc4(x)
            return x
    
class PoissonNet:
    """
    This class is a blueprint for solving a 1D Heat equation with Dirichlet BC
    """
    def __init__(self, model, device):
        self.device=device
        self.model = model.to(self.device)
        self.mseloss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def training(self, X_int_train,X_bc_train,x_int_test,y_bc_train, y_int_test, epochs):
        res = pd.DataFrame(None, columns=['Training Loss', 'Test Loss'], dtype=float)
        for e in range(epochs):
            self.model.train()

            self.optimizer.zero_grad()
            y_bc_pred = self.model(X_bc_train)
            loss_bc = self.mseloss(y_bc_pred, y_bc_train)
            
            u = self.model(X_int_train)
            du_dx = torch.autograd.grad(inputs=X_int_train, outputs=u, grad_outputs=torch.ones_like(u).to(self.device),retain_graph = True, create_graph=True)[0]
            du_dxx = torch.autograd.grad(inputs=X_int_train, outputs=du_dx, grad_outputs=torch.ones_like(du_dx).to(self.device), retain_graph=True, create_graph=True)[0][:, 0]
            #print(x.shape, u.squeeze().shape, du_dx.shape, du_dxx.shape)
            loss_pde = self.mseloss(-du_dxx, (np.pi**2)*torch.sin(np.pi*X_int_train.squeeze()))
            
            loss = loss_pde + loss_bc
            res.loc[e, 'Training Loss'] = torch.sum(loss.item())
            loss.backward(retain_graph=True)
            self.optimizer.step()
     
            self.model.eval() 
            with torch.no_grad():
                y_test_pred = self.model(x_int_test)
                test_loss = self.mseloss(y_test_pred, y_int_test)
                res.loc[e, 'Test Loss'] = test_loss.item()
        return res


class ReadyNet:
    """
    A blueprint to create a physics-informed neural network to solve a 1D Reaction-diffusion equation.    
    """
    def __init__(self, model, device):
        self.device=device
        self.model = model.to(self.device)
        self.mseloss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())


    def training(self, X_int_train,X_bic_train, X_int_test, y_bic_train,y_int_test, epochs):
        res = pd.DataFrame(None, index=range(epochs), columns=['Training Loss', 'Test Loss'], dtype=float)
        for e in range(epochs):
            self.model.train()

            self.optimizer.zero_grad()
            y_bic_pred = self.model(X_bic_train)
            loss_data = self.mseloss(y_bic_pred, y_bic_train)

            u = self.model(X_int_train)

            du_dX = torch.autograd.grad(inputs=X_int_train, outputs=u, grad_outputs=torch.ones_like(u).to(self.device), retain_graph=True, create_graph=True)[0]
            du_dt = du_dX[:, 1]
            du_dxx = torch.autograd.grad(inputs=X_int_train, outputs=du_dX, grad_outputs=torch.ones_like(du_dX).to(self.device), retain_graph=True, create_graph=True)[0][:, 0]
            # print(u.shape, du_dt.shape, du_dx.shape, du_dxx.shape)
            loss_pde = self.mseloss(du_dt, du_dxx + 1.5*torch.sin(2*X_int_train[:, 1]))
            
            loss = loss_pde + loss_data
            res.loc[e, 'Training Loss'] = loss.item()
            loss.backward(retain_graph=True)
            self.optimizer.step()
     
            self.model.eval() 
            with torch.no_grad():
                y_val_pred = self.model(X_int_test) #TODO: implement PDE-loss if needed
                test_loss = self.mseloss(y_val_pred, y_int_test)
                res.loc[e, 'Test Loss'] = test_loss.item()
        return res

class NSNet:
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(self.device)
        self.mseloss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def training(self, X_int_train,X_bic_train, y_bic_train, epochs, optimizer):
        res = pd.DataFrame(None, columns=['Training Loss'], dtype=float)
        for e in range(epochs):
            self.model.train()

            optimizer.zero_grad()
            # order: t, x, y, z
            y_bic_pred = self.model(X_bic_train)
            loss_bic = self.mseloss(y_bic_pred, y_bic_train)

            
            u = self.model(X_int_train)
            loss = NSPDE(X_int_train,u, self.device)
            res.loc[e, 'Training Loss'] = loss.item()
            loss.backward()
            optimizer.step()
        return res
    

class BurgersNet:
    def __init__(self, model, device):
        self.device=device
        self.model = model.to(self.device)
        self.mseloss = torch.nn.MSELoss()
    
    def training(self, X_int_train, X_bc_train, X_ic_train, y_bc_train,y_ic_train, epochs, optimizer):
        res = pd.DataFrame(None, columns=['Training Loss'], dtype=float)
        for e in range(epochs):
            self.model.train()

            optimizer.zero_grad()

            y_bc_pred = self.model(X_bc_train)
            loss_bc = self.mseloss(y_bc_pred, y_bc_train)

            y_ic_pred = self.model(X_ic_train)
            loss_ic = self.mseloss(y_ic_pred, y_ic_train)

            u = self.model(X_int_train)
            loss_pde = BurgersPDE(X_int_train, u, self.device)
            
            loss = loss_pde + loss_bc + loss_ic
            res.loc[e, 'Training Loss'] = loss.item()
            loss.backward()
            self.optimizer.step()
            '''
            self.model.eval() 
            with torch.no_grad():
                y_test_pred = self.model(x_int_test)
                test_loss = self.mseloss(y_test_pred, y_int_test)
                res.loc[e, 'Test Loss'] = test_loss.item()
            '''
        return res    