import os

from models import *

def main(pde:str,
         gamma_1:float,
         gamma_2:float,
         hidden_units_1:int,
         hidden_units_2:int,
         lbfgs_epochs:int,
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
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, h_1={hidden_units_1}, h_2={hidden_units_2}, epochs={lbfgs_epochs}")
    else:
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, g_3={gamma_3}, h_1={hidden_units_1}, h_2={hidden_units_2}, h_3={hidden_units_3}, epochs = {lbfgs_epochs}")
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
    print(f"Model: {net.model}")
    
    if not sampler:
        h = 0.01
        x = torch.arange(-1, 1 + h, h)
        t = torch.arange(0, 1 + h, h)
        X = torch.stack(torch.meshgrid(x[1:-2], t[1:-2], indexing='ij')).reshape(2, -1).T

        bc1 = torch.stack(torch.meshgrid(x[0],
                                         t,
                                         indexing='ij')).reshape(2, -1).T
        bc2 = torch.stack(torch.meshgrid(x[-1],
                                         t,
                                         indexing='ij')).reshape(2, -1).T
        ic = torch.stack(torch.meshgrid(x,
                                        t[0],
                                        indexing='ij')).reshape(2, -1).T
        X_bc_train = torch.cat([bc1, bc2])
        y_bc1 = torch.zeros(len(bc1))
        y_bc2 = torch.zeros(len(bc2))
        y_ic = -torch.sin(np.pi * ic[:, 0]).unsqueeze(1)
        y_bc_train = torch.cat([y_bc1, y_bc2])
        y_bc_train = y_bc_train.unsqueeze(1)
    
        X = X.to(device)
        X_bc_train = X_bc_train.to(device)
        y_bc_train = y_bc_train.to(device)
        ic = ic.to(device)
        y_ic = y_ic.to(device)
        X.requires_grad = True
        results = net.training(X_int_train=X,
                               X_bc_train=X_bc_train,
                               X_ic_train=ic,
                               y_bc_train=y_bc_train,
                               y_ic_train=y_ic,
                               lbfgs_epochs=lbfgs_epochs
                  )

        if not gamma_3:
            file_name = generate_file_name(pde=pde,
                                           epochs=lbfgs_epochs,
                                           hidden_units_1=hidden_units_1,
                                           hidden_units_2=hidden_units_2,
                                           gamma_1=gamma_1,
                                           gamma_2=gamma_2
            )
            place = f'results/{pde}/2layer/normalized/'
            results_directory = os.path.join(directory, place)
        else:
            file_name = generate_file_name(pde=pde,
                                           epochs=lbfgs_epochs,
                                           hidden_units_1=hidden_units_1,
                                           hidden_units_2=hidden_units_2,
                                           gamma_1=gamma_1,
                                           gamma_2=gamma_2,
                                           hidden_units_3=hidden_units_3,
                                           gamma_3=gamma_3
            )
            place = f'results/{pde}/3layer/normalized/'
            results_directory = os.path.join(directory, place)
        save_results(results=results,
                 directory=results_directory,
                 file_name=file_name
    )
        path = os.path.join(results_directory, file_name) + '_model.pth'
        torch.save(net.model.state_dict(), path)

    else:
        full_space = [torch.Tensor([-1., 0.]), torch.Tensor([1., 1.])]
        X_int_train = data_gen(space=full_space, n_samples=2000, sampler=sampler).to(device)
        X_int_train.requires_grad=True
    
        bc1 = torch.stack(torch.meshgrid(torch.Tensor([-1.]),
                                         data_gen(space=[torch.Tensor([0., 1.])], n_samples=100, sampler=sampler).squeeze(), indexing='ij')).reshape(2, -1).T
        bc2 = torch.stack(torch.meshgrid(torch.Tensor([1.]), data_gen(space=[torch.Tensor([0., 1.])], n_samples=100, sampler=sampler).squeeze(), indexing='ij')).reshape(2, -1).T
        X_ic_train = torch.stack(torch.meshgrid(data_gen(space=[torch.Tensor([-1.,1.])], n_samples=100, sampler=sampler).squeeze(), torch.Tensor([0.]), indexing='ij')).reshape(2, -1).T.to(device)
        X_bc_train = torch.cat([bc1, bc2]).to(device)

        y_bc1 = torch.zeros(len(bc1))
        y_bc2 = torch.zeros(len(bc2))
        y_bc_train = torch.cat([y_bc1, y_bc2]).unsqueeze(1).to(device)
        y_ic_train = -torch.sin(np.pi*X_ic_train[:, 0]).unsqueeze(1).to(device)

        results = net.training(X_int_train=X_int_train, X_bc_train=X_bc_train, X_ic_train=X_ic_train, y_bc_train=y_bc_train,y_ic_train=y_ic_train,lbfgs_epochs=lbfgs_epochs)

    # Save accuracy results
        if not gamma_3:
            file_name = generate_file_name(pde=pde,
                                           epochs=lbfgs_epochs,
                                           hidden_units_1=hidden_units_1,
                                           hidden_units_2=hidden_units_2,
                                           gamma_1=gamma_1,
                                           gamma_2=gamma_2
                        )
            place = f'results/{pde}/2layer/{sampler}/'
            results_directory = os.path.join(directory, place)
        else:
            file_name = generate_file_name(pde=pde,
                                           epochs=lbfgs_epochs,
                                           hidden_units_1=hidden_units_1,
                                           hidden_units_2=hidden_units_2,
                                           gamma_1=gamma_1,
                                           gamma_2=gamma_2,
                                           hidden_units_3=hidden_units_3,
                                           gamma_3=gamma_3
                        )
            place = f'results/{pde}/3layer/{sampler}/'
            results_directory = os.path.join(directory, place)
        save_results(results=results,
                     directory=results_directory,
                     file_name=file_name
        )
        path = os.path.join(results_directory, file_name) + '_model.pth'
        torch.save(net.model.state_dict(), path)

    return

if __name__ == '__main__':
    pde='Burgers'
    gamma_1_list = [0.5, 0.7, 1.0]
    gamma_2_list = [0.5, 0.7, 1.0]
    #gamma_3_list = [0.5, 0.7, 1.0]
    hidden_units_1=100
    hidden_units_2=100
    #hidden_units_3=100
    lbfgs_epochs = 25000
    sampler_list = ['random','LHS', 'Halton', 'Sobol']
    directory=os.getcwd()
    for gamma_1 in gamma_1_list:
        for gamma_2 in gamma_2_list:
            for de in sampler_list:
                main(pde=pde,gamma_1=gamma_1,
                     gamma_2=gamma_2,
                     hidden_units_1=hidden_units_1,
                     hidden_units_2=hidden_units_2,
                     lbfgs_epochs=lbfgs_epochs,
                     directory=directory,
                     sampler=de
                )