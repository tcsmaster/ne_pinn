import os
from helpers import *
from models import *

def main(pde,
         gamma_1,
         gamma_2,
         hidden_units_1,
         hidden_units_2,
         epochs,
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
    print(f"PDE:3D Navier-Stokes")
    if (not gamma_3):
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, h_1={hidden_units_1}, h_2={hidden_units_2}, epochs={epochs}")
    else:
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, g_3={gamma_3}, h_1={hidden_units_1}, h_2={hidden_units_2}, h_3={hidden_units_3}, epochs = {epochs}")

    if (not gamma_3):
        net = NSNet(MLP2(num_input=4,num_output=4,hidden_units_1=hidden_units_1, hidden_units_2=hidden_units_2, gamma_1=gamma_1, gamma_2=gamma_2))
    else:
        net = NSNet(MLP3(num_input=4,num_output=4,hidden_units_1=hidden_units_1, hidden_units_2=hidden_units_2, hidden_units_3=hidden_units_3, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3))
    print(f"Model: {net.model}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    h = 0.01
    x = torch.arange(0, 1 + h, h, dtype=torch.float32)
    y = torch.arange(0, 1 + h, h, dtype=torch.float32)
    z = torch.arange(0, 1 + h, h, dtype=torch.float32)
    t = torch.arange(0, 1 + h, h, dtype=torch.float32)
        # exact solution
    X = torch.stack(torch.meshgrid(t[1:-2],x[1:-2],y[1:-2],z[1:-2], indexing='ij')).reshape(4, -1).T.to(device)
    X.requires_grad = True
    
    # training data
    bc1 = torch.stack(torch.meshgrid(t, x[0], y, z, indexing='ij')).reshape(4, -1).T
    bc2 = torch.stack(torch.meshgrid(t, x[-1], y, z, indexing='ij')).reshape(4, -1).T
    bc3 = torch.stack(torch.meshgrid(t, x, y[0], z, indexing='ij')).reshape(4, -1).T
    bc4 = torch.stack(torch.meshgrid(t, x, y[-1], z, indexing='ij')).reshape(4, -1).T
    bc5 = torch.stack(torch.meshgrid(t, x, y, z[0], indexing='ij')).reshape(4, -1).T
    bc6 = torch.stack(torch.meshgrid(t, x, y, z[-1], indexing='ij')).reshape(4, -1).T
    ic1 = torch.stack(torch.meshgrid(t[0], x, y, z, indexing='ij')).reshape(4, -1).T
    ic2 = torch.stack(torch.meshgrid(t[1], x, y, z, indexing='ij')).reshape(4, -1).T
    X_bic_train = torch.cat([bc1, bc2, bc3,bc4,bc5,bc6,ic1,ic2]).to(device)
    y_bic_train = torch.cat(u_func(X_bic_train),v_func(X_bic_train),w_func(X_bic_train),p_func(X_bic_train), dim=1).to(device)
    '''
    full_space = [torch.Tensor([-1., 0.]), torch.Tensor([1., 1.])]
    X_int_train = data_gen(space=full_space, n_samples=2000, sampler=sampler).to(device)
    X_int_train.requires_grad=True
    
    bc1 = torch.stack(torch.meshgrid(torch.Tensor([-1.]), data_gen(space=[torch.Tensor([0., 1.])], n_samples=100, sampler=sampler).squeeze(), indexing='ij')).reshape(2, -1).T
    bc2 = torch.stack(torch.meshgrid(torch.Tensor([1.]), data_gen(space=[torch.Tensor([0., 1.])], n_samples=100, sampler=sampler).squeeze(), indexing='ij')).reshape(2, -1).T
    X_ic_train = torch.stack(torch.meshgrid(data_gen(space=[torch.Tensor([-1.,1.])], n_samples=100, sampler=sampler).squeeze(), torch.Tensor([0.]), indexing='ij')).reshape(2, -1).T.to(device)
    X_bc_train = torch.cat([bc1, bc2]).to(device)

    y_bc1 = torch.zeros(len(bc1))
    y_bc2 = torch.zeros(len(bc2))
    y_bc_train = torch.cat([y_bc1, y_bc2]).unsqueeze(1).to(device)
    y_ic_train = -torch.sin(np.pi*X_ic_train[:, 0]).unsqueeze(1).to(device)
    results = net.training(X_int_train=X_int_train, X_bc_train=X_bc_train, X_ic_train=X_ic_train, y_bc_train=y_bc_train,y_ic_train=y_ic_train, epochs=epochs)
    '''
    results = net.training(X_int_train=X,X_bic_train= X_bic_train, y_bic_train= y_bic_train, epochs=epochs)

    # Save accuracy results
    if not gamma_3:
        file_name = generate_file_name(pde=pde,epochs=epochs,
                                       hidden_units_1=hidden_units_1,
                                       hidden_units_2=hidden_units_2,
                                       gamma_1=gamma_1,gamma_2=gamma_2)
        place = f'results\\Burgers\\2layer\\normalized\\'

    else:
        file_name =generate_file_name(pde=pde,epochs=epochs,
                                      hidden_units_1=hidden_units_1,
                                   hidden_units_2=hidden_units_2,
                                   gamma_1=gamma_1,
                                   gamma_2=gamma_2,
                                   hidden_units_3=hidden_units_3,
                                   gamma_3=gamma_3
        )
        place = f'results\\Burgers\\3layer\\normalized\\'
        results_directory = os.path.join(directory, place)
    save_results(results=results,
                 directory=results_directory,
                 file_name=file_name
    )
    path = os.path.join(results_directory, file_name) + '_model.pth'
    torch.save(net.model.state_dict(), path)

    return

if __name__ == '__main__':
    pde='Navier-Stokes'
    gamma_1 = 0.5
    gamma_2 = 0.5
    #gamma_3_list = [0.5, 0.7, 1.0]
    hidden_units_1=100
    hidden_units_2=100
    #hidden_units_3=100
    epochs=10000
    sampler_list = ['random','LHS', 'Halton', 'Sobol']
    directory=os.getcwd()
    for sampler in sampler_list:
        main(pde=pde,gamma_1=gamma_1,
                     gamma_2=gamma_2,
                     hidden_units_1=hidden_units_1,
                     hidden_units_2=hidden_units_2,
                     epochs=epochs,
                     directory=directory,
                     sampler=sampler
                )