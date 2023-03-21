import os
from helpers import *
from models import *
from math import pi

"""Trains a neural network model on a dataset and saves the resulting 
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
    
def main(pde,
         gamma_1,
         gamma_2,
         hidden_units_1,
         hidden_units_2,
         epochs,
         directory,
         sampler,
         gamma_3 = None,
         hidden_units_3 = None,
    ):
    print(f"PDE:{pde}")
    if (not gamma_3):
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, h_1={hidden_units_1}, h_2={hidden_units_2}, epochs={epochs}")
    else:
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, g_3={gamma_3}, h_1={hidden_units_1}, h_2={hidden_units_2}, h_3={hidden_units_3}, epochs = {epochs}")

    # training data

    if (not gamma_3):
        net = ReadyNet(MLP2(num_input=2,num_output=1,hidden_units_1=hidden_units_1, hidden_units_2=hidden_units_2, gamma_1=gamma_1, gamma_2=gamma_2))
    else:
        net = ReadyNet(MLP3(num_input=2,num_output=1,hidden_units_1=hidden_units_1, hidden_units_2=hidden_units_2, hidden_units_3=hidden_units_3, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3))
    print(f"Model: {net.model}")

    full_space =  [torch.Tensor([-pi, 0.]), torch.Tensor([pi, 1.])]
    X_int_train = data_gen(space=full_space, n_samples=128, sampler=sampler)
    X_int_train.requires_grad=True
    x_int_test = data_gen(space=full_space, n_samples=30, sampler='random')
    y_int_test = torch.sin(x_int_test[:, 0])*torch.exp(-x_int_test[:, 1])
    y_int_test = y_int_test.unsqueeze(1)

    bc1 = torch.stack(torch.meshgrid(torch.Tensor([-pi]), data_gen(space=[torch.Tensor([0., 1.])], n_samples=20, sampler=sampler).squeeze(), indexing='ij')).reshape(2, -1).T
    bc2 = torch.stack(torch.meshgrid(torch.Tensor([pi]), data_gen(space=[torch.Tensor([0., 1.])], n_samples=20, sampler=sampler).squeeze(), indexing='ij')).reshape(2, -1).T
    ic = torch.stack(torch.meshgrid(data_gen(space=[torch.Tensor([-pi, pi])], n_samples=20, sampler=sampler).squeeze(), torch.Tensor([0.]), indexing='ij')).reshape(2, -1).T
    X_bic_train = torch.cat([bc1, bc2, ic]) #TODO: separate this to have diff. ic and bc points

    y_bc1 = torch.zeros(len(bc1))
    y_bc2 = torch.zeros(len(bc2))
    y_ic = torch.sin(ic[:, 0])
    y_bic_train = torch.cat([y_bc1, y_bc2, y_ic]).unsqueeze(1)
    results = net.training(X_int_train,X_bic_train, x_int_test, y_bic_train,y_int_test, epochs)

    if not gamma_3:
        file_name = generate_file_name(pde=pde,
                                   epochs=epochs,
                                   hidden_units_1=hidden_units_1,
                                   hidden_units_2=hidden_units_2,
                                   gamma_1=gamma_1,
                                   gamma_2=gamma_2
        )
        results_directory = os.path.join(directory, f'results\\{pde}\\2layer\\{sampler}')
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
        results_directory = os.path.join(directory, f'results\\{pde}\\3layer\\{sampler}')
    save_results(results=results,
                directory=results_directory,
                 file_name=file_name
    )

    return

if __name__ == '__main__':
    pde='Reaction-diffusion'
    gamma_1 = 0.5
    gamma_2 = 0.5
    hidden_units_1=100
    hidden_units_2=100
    epochs=4000
    directory=os.getcwd()
    sampler_list = ['random','Halton','LHS', 'Sobol']
    for sampler in sampler_list:
        main(pde=pde,
             gamma_1=gamma_1,
             gamma_2=gamma_2,
             hidden_units_1=hidden_units_1,
             hidden_units_2=hidden_units_2,
             epochs=epochs,
             directory=directory,
             sampler=sampler
        )
