import os
from utils import *
from models import *
from torch.optim import Adam
from pdes import NSPDE


class NSNet:
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(self.device)
        
    
    def training(self,
                 X_int_train,
                 X_bic_train,
                 y_bic_train,
                 adam_epochs
        ):
        res = pd.DataFrame(None, columns=['Training Loss'], dtype=float)
        optimizer = Adam(self.model.parameters(), amsgrad=True)
        self.model.train()
        for e in range(adam_epochs):
            optimizer.zero_grad()
            u = self.model(X_int_train)
            loss_pde = NSPDE(X_int_train, u, self.device)
            y_bic_pred = self.model(X_bic_train)
            loss_bic = torch.nn.MSELoss()(y_bic_pred, y_bic_train)               
            loss = loss_pde + loss_bic
            loss.backward(retain_graph=True)
            optimizer.step()
            res.loc[e, 'Training Loss'] = loss_pde.item()
        return res

def main(pde,
         gamma_1,
         gamma_2,
         hidden_units_1,
         hidden_units_2,
         adam_epochs,
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
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, h_1={hidden_units_1}, h_2={hidden_units_2}, epochs={adam_epochs}")
    else:
        print(f"Parameters: g_1={gamma_1}, g_2={gamma_2}, g_3={gamma_3}, h_1={hidden_units_1}, h_2={hidden_units_2}, h_3={hidden_units_3}, epochs = {adam_epochs}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if (not gamma_3):
        net = NSNet(MLP2(num_input=4,
                         num_output=4,
                         hidden_units_1=hidden_units_1,
                         hidden_units_2=hidden_units_2,
                         gamma_1=gamma_1,
                         gamma_2=gamma_2,
                         sampler=sampler
                    ),
                    device=device)
    else:
        net = NSNet(MLP3(num_input=4,
                         num_output=4,
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
        h = 0.1
        x = torch.arange(0,
                         1 + h,
                         h,
                         dtype=torch.float32, device=device, requires_grad=True)
        y = torch.arange(0,
                         1 + h,
                         h,
                         dtype=torch.float32, device=device, requires_grad=True)
        z = torch.arange(0,
                         1 + h,
                         h,
                         dtype=torch.float32, device=device, requires_grad=True)
        t = torch.arange(0,
                         1 + h,
                         h,
                         dtype=torch.float32, device=device, requires_grad=True)
            # exact solution
        X_int_train = torch.stack(torch.meshgrid(t[1:-2],
                                                 x[1:-2],
                                                 y[1:-2],
                                                 z[1:-2],
                                                 indexing='ij')).reshape(4, -1).T

    # training data
        bc1 = torch.stack(torch.meshgrid(t,
                                       x[0],
                                       y,
                                       z,
                                       indexing='ij')).reshape(4, -1).T
        bc2 = torch.stack(torch.meshgrid(t,
                                       x[-1],
                                       y,
                                       z,
                                       indexing='ij')).reshape(4, -1).T
        bc3 = torch.stack(torch.meshgrid(t,
                                       x,
                                       y[0],
                                       z,
                                       indexing='ij')).reshape(4, -1).T
        bc4 = torch.stack(torch.meshgrid(t,
                                       x,
                                       y[-1],
                                       z,
                                       indexing='ij')).reshape(4, -1).T
        bc5 = torch.stack(torch.meshgrid(t,
                                       x,
                                       y,
                                       z[0],
                                       indexing='ij')).reshape(4, -1).T
        bc6 = torch.stack(torch.meshgrid(t,
                                       x,
                                       y,
                                       z[-1],
                                       indexing='ij')).reshape(4, -1).T
        ic1 = torch.stack(torch.meshgrid(t[0],
                                       x,
                                       y,
                                       z,
                                       indexing='ij')).reshape(4, -1).T
        ic2 = torch.stack(torch.meshgrid(t[-1],
                                       x,
                                       y,
                                       z,
                                       indexing='ij')).reshape(4, -1).T
        X_bic_train = torch.cat([bc1, bc2, bc3,bc4,bc5,bc6,ic1,ic2])
        y_bic_train = torch.cat((u_func(X_bic_train),
                               v_func(X_bic_train),
                               w_func(X_bic_train),
                               p_func(X_bic_train)),
                               dim=1)
        results = net.training(X_int_train=X_int_train,
                               X_bic_train=X_bic_train,
                               y_bic_train=y_bic_train,
                               adam_epochs=adam_epochs)
        if not gamma_3:
            file_name = generate_file_name(pde=pde,
                                       epochs=adam_epochs,
                                       hidden_units_1=hidden_units_1,
                                       hidden_units_2=hidden_units_2,
                                       gamma_1=gamma_1,
                                       gamma_2=gamma_2
                      )
            place = f'results/{pde}/2layer/normalized/Adam_with_amsgrad'
            results_directory = os.path.join(directory, place)
        else:
            file_name =generate_file_name(pde=pde,
                                       epochs=adam_epochs,
                                       hidden_units_1=hidden_units_1,
                                       hidden_units_2=hidden_units_2,
                                       gamma_1=gamma_1,
                                       gamma_2=gamma_2,
                                       hidden_units_3=hidden_units_3,
                                       gamma_3=gamma_3
                    )
            place = f'results/{pde}/3layer/normalized/Adam_with_amsgrad'
            results_directory = os.path.join(directory, place)
        save_results(results=results,
                   directory=results_directory,
                   file_name=file_name
                   )
        path = os.path.join(results_directory, file_name) + '_model.pth'
        torch.save(net.model.state_dict(), path)
    else:
        full_space = [torch.tensor([0., 0., 0., 0.]), torch.tensor([1., 1., 1., 1.])]
        X_int_train = data_gen(space=full_space, n_samples = 4000, sampler=sampler).to(device)
        X_int_train.requires_grad = True

        X_bic_train = []

        data1 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data2 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data3 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        bc1= torch.stack(torch.meshgrid(data1,torch.tensor([0.]), data2, data3, indexing="ij")).reshape(4, -1).T
        X_bic_train.append(bc1)

        data1 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data2 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data3 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        bc2= torch.stack(torch.meshgrid(data1, torch.tensor([1.]), data2, data3, indexing="ij")).reshape(4, -1).T
        X_bic_train.append(bc2)

        data1 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data2 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data3 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        bc3= torch.stack(torch.meshgrid(data1, data2, torch.tensor([0.]), data3, indexing="ij")).reshape(4, -1).T
        X_bic_train.append(bc3)

        data1 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data2 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data3 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        bc4= torch.stack(torch.meshgrid(data1, data2, torch.tensor([1.]), data3, indexing="ij")).reshape(4, -1).T
        X_bic_train.append(bc4)

        data1 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data2 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data3 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        bc5= torch.stack(torch.meshgrid(data1, data2, data3, torch.tensor([0.]), indexing="ij")).reshape(4, -1).T
        X_bic_train.append(bc5)

        data1 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data2 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data3 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        bc6= torch.stack(torch.meshgrid(data1, data2, data3, torch.tensor([1.]),indexing="ij")).reshape(4, -1).T
        X_bic_train.append(bc6)

        data1 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data2 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data3 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        ic1= torch.stack(torch.meshgrid(torch.tensor([0.]), data1, data2, data3, indexing="ij")).reshape(4, -1).T
        X_bic_train.append(ic1)

        data1 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data2 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        data3 = data_gen(space=[torch.tensor([0.]), torch.tensor([1.])], n_samples=10, sampler = sampler).squeeze()
        ic2= torch.stack(torch.meshgrid(torch.tensor([1.]),data1, data2, data3, indexing="ij")).reshape(4, -1).T
        X_bic_train.append(ic2)

        X_bic_train = torch.cat(X_bic_train, dim=0)
        y_bic_train = torch.cat((u_func(X_bic_train),v_func(X_bic_train),w_func(X_bic_train),p_func(X_bic_train)), dim=1).to(device)
        results = net.training(X_int_train=X_int_train,X_bic_train= X_bic_train, y_bic_train= y_bic_train, epochs=epochs)
        if not gamma_3:
            file_name = generate_file_name(pde=pde,epochs=epochs,
                                       hidden_units_1=hidden_units_1,
                                       hidden_units_2=hidden_units_2,
                                       gamma_1=gamma_1,gamma_2=gamma_2)
            place = f'results/{pde}/2layer/{sampler}/'
            results_directory = os.path.join(directory, place)
        else:
            file_name =generate_file_name(pde=pde,epochs=epochs,
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
    pde='Navier-Stokes'
    gamma_1_list = [0.5, 0.7, 1.0]
    gamma_2_list = [0.5, 0.7, 1.0]
    gamma_3_list = [0.5, 0.7, 1.0]
    hidden_units_1=100
    hidden_units_2=100
    hidden_units_3=100
    adam_epochs=20000
    directory=os.getcwd()
    for gamma_1 in gamma_1_list:
        for gamma_2 in gamma_2_list:
            main(pde=pde,gamma_1=gamma_1,
                     gamma_2=gamma_2,
                     hidden_units_1=hidden_units_1,
                     hidden_units_2=hidden_units_2,
                     adam_epochs=adam_epochs,
                     directory=directory
                )