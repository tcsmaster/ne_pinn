import torch
from torch.nn import MSELoss
from numpy import pi


def PoissonPDE(x, u, device):
    du_dx = torch.autograd.grad(
        inputs=x,
        outputs=u,
        grad_outputs=torch.ones_like(
            u,
            device=device
        ),
        create_graph=True,
        retain_graph=True
    )[0]
    du_dxx = torch.autograd.grad(
        inputs=x,
        outputs=du_dx,
        grad_outputs=torch.ones_like(
            du_dx,
            device=device
        ),
        retain_graph=True,
        create_graph=True
    )[0][:, 0:1]
    loss = MSELoss()(-du_dxx, (pi**2)*torch.sin(pi*x))
    return loss

def BurgersPDE(x, u, device):
    du_dX = torch.autograd.grad(
        x,
        u,
        torch.ones_like(u,device=device),
        create_graph=True,
        retain_graph=True
    )[0]
    du_dx = du_dX[:,0:1]
    du_dt = du_dX[:,1:2]
    du_dxx = torch.autograd.grad(
        x,
        du_dx,
        torch.ones_like(du_dx,device=device),
        retain_graph=True,
        create_graph=True
    )[0][:, 0:1]
    loss = MSELoss()(du_dt + du_dx*u,0.01/pi*du_dxx)
    return loss