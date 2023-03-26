import os
from helpers import *
from models import *
from torch import device

def main(pde,
         gamma_1,
         gamma_2,
         hidden_units_1,
         hidden_units_2,
         epochs,
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
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, h_1={hidden_units_1}, h_2={hidden_units_2}, epochs={epochs}")
    else:
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, g_3={gamma_3}, h_1={hidden_units_1}, h_2={hidden_units_2}, h_3={hidden_units_3}, epochs = {epochs}")

    if (not gamma_3):
        net = PoissonNet(MLP2(num_input=1,num_output=1,hidden_units_1=hidden_units_1, hidden_units_2=hidden_units_2, gamma_1=gamma_1, gamma_2=gamma_2, sampler=sampler))
    else:
        net = PoissonNet(MLP3(num_input=1,num_output=1,hidden_units_1=hidden_units_1, hidden_units_2=hidden_units_2, hidden_units_3=hidden_units_3, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3, sampler=sampler))
    print(f"Model: {net.model}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    full_space = [(-1., 1.)]
    #X_int_train = data_gen(space=full_space, n_samples=128, sampler=sampler)
    X_int_train = torch.arange(-0.9, 1.1, 0.1).reshape(1, -1).T
    X_int_train.requires_grad=True
    x_int_test = data_gen(space=full_space, n_samples=30, sampler='random')
    y_int_test = torch.sin(np.pi*x_int_test)

    bc1 = torch.Tensor([-1.])
    bc2 = torch.Tensor([1.])
    X_bc_train = torch.cat([bc1, bc2]).unsqueeze(1)

    y_bc1 = torch.zeros(len(bc1))
    y_bc2 = torch.zeros(len(bc2))
    y_bc_train = torch.cat([y_bc1, y_bc2]).unsqueeze(1)
    results = net.training(X_int_train,X_bc_train, x_int_test, y_bc_train,y_int_test, epochs)



    # Save accuracy results
    if not gamma_3:
        file_name = generate_file_name(pde=pde,
                                       epochs=epochs,
                                       hidden_units_1=hidden_units_1,
                                       hidden_units_2=hidden_units_2,
                                       gamma_1=gamma_1,
                                       gamma_2=gamma_2
        )
        results_directory = os.path.join(directory, f'results/{pde}/2layer/normalized/')
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
    gamma_1_list = [0.5,0.6,0.7,0.8,0.9,1.0]
    gamma_2_list = [0.5,0.6,0.7,0.8,0.9,1.0]
    #gamma_3 = 0.5
    hidden_units_1=100
    hidden_units_2=100
    #hidden_units_3=100
    epochs=10000
    directory=os.getcwd()
    #sampler_list = ['random', 'Halton', 'LHS', 'Sobol']
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