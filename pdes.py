import torch
from torch.nn import MSELoss
from numpy import pi


def PoissonPDE(x, u, device):
    """
    Computes the residual u_xx + pi²*sin(pi*x)

    Arguments:
    ----------
        x: torch.tensor
            Tensor of PINN input data
        u: torch.tensor
            Tensor of PINN output for x input
        device:
            Device, where the computation happens. GPU if available, CPU otherwise
    """
    du_dx = torch.autograd.grad(
        u,
        x,
        torch.ones_like(u,device=device),
        create_graph=True,
        retain_graph=True
    )[0]
    # retain the comp. graph so second derivative can be computed
    # and loss can be backpropped
    du_dxx = torch.autograd.grad(
        du_dx,
        x,
        torch.ones_like(du_dx,device=device),
        retain_graph=True,
        create_graph=True
    )[0][:, 0:1]
    # calculate the mse loss for Poisson PDE
    loss = MSELoss()(-du_dxx, (pi**2)*torch.sin(pi*x))
    return loss

def BurgersPDE(x, u, device):
    """
    Computes the residual u_t + u*u_x - 0.01/pi * u_xx

    Arguments:
    ----------
        x: float
            PINN input data. Has dimensions (num_points x 2), first column is the x,
            second column is the t coordinate.
        u: float
            PINN output for x input
        device:
            device, where the computation happens. GPU if available, CPU otherwise
    """
    du_dX = torch.autograd.grad(
        u,
        x,
        torch.ones_like(u,device=device),
        create_graph=True,
        retain_graph=True
    )[0]
    # du_dX has the same dimensions as x
    du_dx = du_dX[:,0:1]
    du_dt = du_dX[:,1:2]
    # retain the comp. graph so second derivative can be computed
    # and loss can be backpropped
    du_dxx = torch.autograd.grad(
        du_dx,
        x,
        torch.ones_like(du_dx,device=device),
        retain_graph=True,
        create_graph=True
    )[0][:, 0:1]
    # Calculate the MSE loss for the Burgers' equation
    loss = MSELoss()(du_dt + du_dx*u,0.01/pi*du_dxx)
    return loss