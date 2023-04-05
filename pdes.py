import torch
from torch.nn import MSELoss
from numpy import pi
def NSPDE(x, u, device):
    u_vel,v_vel,w_vel,p=u[:, 0:1],u[:, 1:2],u[:, 2:3],u[:, 3:4]
    du_dX = torch.autograd.grad(inputs=x,
                                outputs=u_vel,
                                grad_outputs=torch.ones_like(u_vel).to(device),
                                create_graph=True
            )[0]
    u_vel_t,u_vel_x,u_vel_y,u_vel_z = du_dX[:, 0],du_dX[:, 1],du_dX[:, 2],du_dX[:, 3]
    u_vel_xx=torch.autograd.grad(inputs=x,
                                 outputs=u_vel_x,
                                 grad_outputs=torch.ones_like(u_vel_x).to(device),
                                 retain_graph = True
             )[0][:, 1]
    u_vel_yy=torch.autograd.grad(inputs=x,
                                 outputs=u_vel_y,
                                 grad_outputs=torch.ones_like(u_vel_y).to(device),
                                 retain_graph = True
             )[0][:, 2]
    u_vel_zz=torch.autograd.grad(inputs=x,
                                 outputs=u_vel_z,
                                 grad_outputs=torch.ones_like(u_vel_z).to(device),
                                 retain_graph = True
             )[0][:, 3]

    dv_dX=torch.autograd.grad(inputs=x,
                                outputs=v_vel,
                                grad_outputs=torch.ones_like(v_vel).to(device),
                                create_graph=True
          )[0]
    v_vel_t, v_vel_x, v_vel_y,v_vel_z = dv_dX[:, 0], dv_dX[:, 1], dv_dX[:, 2], dv_dX[:, 3]
    v_vel_xx=torch.autograd.grad(inputs=x,
                                   outputs=v_vel_x,
                                   grad_outputs=torch.ones_like(v_vel_x).to(device),
                                   retain_graph = True
             )[0][:, 1]
    v_vel_yy=torch.autograd.grad(inputs=x,
                                 outputs=v_vel_y,
                                 grad_outputs=torch.ones_like(v_vel_y).to(device),
                                 retain_graph = True
             )[0][:, 2]
    v_vel_zz=torch.autograd.grad(inputs=x,
                                 outputs=v_vel_z,
                                 grad_outputs=torch.ones_like(v_vel_z).to(device),
                                 retain_graph = True
             )[0][:, 3]

    dw_dX=torch.autograd.grad(inputs=x,
                              outputs=w_vel,
                              grad_outputs=torch.ones_like(u_vel).to(device),
                              create_graph=True
          )[0]
    w_vel_t, w_vel_x, w_vel_y,w_vel_z = dw_dX[:, 0], dw_dX[:, 1], dw_dX[:, 2], dw_dX[:, 3]
    w_vel_xx=torch.autograd.grad(inputs=x,
                                   outputs=w_vel_x,
                                   grad_outputs=torch.ones_like(w_vel_x).to(device),
                                   retain_graph=True
             )[0][:, 1]
    w_vel_yy=torch.autograd.grad(inputs=x,
                                 outputs=w_vel_y,
                                 grad_outputs=torch.ones_like(w_vel_y).to(device),
                                 retain_graph = True
             )[0][:, 2]
    w_vel_zz=torch.autograd.grad(inputs=x,
                                 outputs=w_vel_z,
                                 grad_outputs=torch.ones_like(w_vel_z).to(device),
                                 retain_graph = True
             )[0][:, 3]

    p_x=torch.autograd.grad(inputs=x,
                              outputs=p,
                              grad_outputs=torch.ones_like(p).to(device),
                              create_graph=True
        )[0][:, 1]
    p_y=torch.autograd.grad(inputs=x,
                            outputs=p,
                            grad_outputs=torch.ones_like(p).to(device),
                            retain_graph=True
        )[0][:, 2]
    p_z=torch.autograd.grad(inputs=x,
                            outputs=p,
                            grad_outputs=torch.ones_like(p).to(device),
                            retain_graph=True
        )[0][:, 3]
            
    momentum_x = MSELoss()(u_vel_t + u_vel.squeeze() * u_vel_x + v_vel.squeeze() * u_vel_y + w_vel.squeeze() * u_vel_z,\
                      p_x - u_vel_xx - u_vel_yy - u_vel_zz)
    momentum_y = MSELoss()(v_vel_t + u_vel.squeeze() * v_vel_x + v_vel.squeeze() * v_vel_y + w_vel.squeeze() * v_vel_z,\
                      p_y - v_vel_xx - v_vel_yy - v_vel_zz)
    momentum_z = MSELoss()(w_vel_t + u_vel.squeeze() * w_vel_x + v_vel.squeeze() * w_vel_y + w_vel.squeeze() * w_vel_z,\
                      p_z - w_vel_xx - w_vel_yy - w_vel_zz)
    continuity = MSELoss()(u_vel_x + v_vel_y + w_vel_z, torch.zeros_like(u_vel_x))
            
    loss = momentum_x + momentum_y + momentum_z + continuity
    return loss

def BurgersPDE(x, u, device):
    du_dX = torch.autograd.grad(inputs=x,
                                outputs=u,
                                grad_outputs=torch.ones_like(u).to(device),
                                create_graph=True,
            )[0]
    du_dt = du_dX[:,1]
    du_dx = du_dX[:,0]
    du_dxx = torch.autograd.grad(inputs=x,
                                 outputs=du_dx,
                                 grad_outputs=torch.ones_like(du_dx).to(device),
                                 retain_graph=True,
             )[0][:, 0]
    print(du_dxx.shape)
    loss = MSELoss()(du_dt + du_dx*u.squeeze(),0.01/pi*du_dxx)
    return loss

def PoissonPDE(x, u, device):
    du_dx = torch.autograd.grad(inputs=x,
                                outputs=u,
                                grad_outputs=torch.ones_like(u).to(device),
                                create_graph=True
            )[0]
    du_dxx = torch.autograd.grad(inputs=x,
                                 outputs=du_dx,
                                 grad_outputs=torch.ones_like(du_dx).to(device),
                                 retain_graph=True
             )[0][:, 0]
    loss_pde = MSELoss()(-du_dxx, (pi**2)*torch.sin(pi*x.squeeze()))