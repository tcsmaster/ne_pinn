import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import generate_file_name

plt.rcParams.update({           # Matplotlib parameter setting
    "font.monospace": [],
    "figure.figsize": (12,8),
    "axes.labelsize": 20,           
    "font.size": 20,
    "legend.fontsize": 20,  
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    })


def load_loss_for_single_gamma(
    pde,
    epochs,
    hidden_units_1,
    hidden_units_2,
    gamma_1,
    gamma_2,
    directory,
    optimizer
):
    """
    Loads a type of loss for a single pair of gamma_1 and gamma_2 from the directory
    directory/results/pde/optimizer.

    Arguments:
    ----------

    pde: str
        The pde that is to be solved
    """
    fname = generate_file_name(
        pde=pde,
        epochs=epochs,
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        gamma_1=gamma_1,
        gamma_2=gamma_2
    )
    results_folder = f'results/{pde}/{optimizer}/'

    # Create full path to data file, including extension
    path = os.path.join(directory, results_folder, fname) + '.csv'
    
    # Load data file
    data = pd.read_csv(path, index_col=0)

    return data


def load_all_losses(
    pde,
    epochs,
    acc,
    gamma_1_list,
    gamma_2_list, 
    hidden_units_1,
    hidden_units_2,
    directory,
    optimizer
):
    """Returns a DataFrame with either test or train accuracy by epoch for  
    lists of gamma values
    
    Parameters
    ----------
    gamma_1_list: list of floats
        the scaling parameters for the first layer
    gamma_2_list: list of floats
        the scaling parameters for the second layer
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    """

    # Dictionary to store data by different gamma_1
    dict_data = dict()

    # Iterate over list of gamma values and load data
    for gamma_1 in gamma_1_list:
        for gamma_2 in gamma_2_list:
            data = load_loss_for_single_gamma(
                pde=pde,
                epochs=epochs,
                hidden_units_1=hidden_units_1,
                hidden_units_2=hidden_units_2,
                gamma_1=gamma_1,
                gamma_2=gamma_2,
                directory=directory,
                optimizer=optimizer
            )
            dict_data[(gamma_1,gamma_2)] = data[acc]
        
    # Concatenate loss data over gamma values
    results = pd.concat(dict_data, axis=1)
    return results


def run_accuracy_plots(
    pde,
    epochs,
    acc,
    gamma_1_list,
    gamma_2_list,
    hidden_units_1,
    hidden_units_2,
    directory,
    optimizer
):
    """Plots and saves figures of test or train accuracy for lists of multiple 
    gamma values for Multi-layer perceptron with two hidden layers (MLP2)
    
    Parameters
    ----------
    gamma_1_list: list of floats
        the mean-field scaling parameters for the first layer 
    gamma_2_list: list of floats
        the mean-field scaling parameters for the second layer 
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    """

    # Load loss data
    data = load_all_losses(
        pde=pde,
        epochs=epochs,
        acc=acc,
        gamma_1_list=gamma_1_list,
        gamma_2_list=gamma_2_list,
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        directory=directory,
        optimizer=optimizer
    )
    figures_directory = os.path.join(directory, f"figures/{pde}/{acc}/")
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory)
    for gamma_1 in gamma_1_list:
        fig = plt.figure(figsize=(20, 10))
        ax = data[gamma_1].plot()
        if acc == "Training Loss":
            ax.set_title(f'Training loss for $\gamma_1={{{gamma_1}}}$')
        elif acc == "Test mse loss":
            ax.set_title(f'Test MSE-loss for $\gamma_1={{{gamma_1}}}$')
        elif acc == "Test_rel_l2_loss":
            ax.set_title(f'Relative $L^2$-loss for $\gamma_1={{{gamma_1}}}$')
        else:
            raise ValueError("This metric is not available!")
        plt.legend(title='$\gamma_2$', loc='lower center', bbox_to_anchor = [0.5, -0.3], ncols = len(gamma_2_list))
        plt.xlabel('Number of Epochs')
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.grid()

        fname = f'plot_gamma1_{gamma_1}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}'

        fig_path = os.path.join(figures_directory, fname)       
        ax.figure.savefig(fig_path + '.jpg', dpi=300, bbox_inches='tight')
        plt.close('all')      
        
    for gamma_2 in gamma_2_list:
        fig = plt.figure(figsize=(20, 10))
        ax = data.xs(gamma_2, level=1, axis=1).plot()
        if acc == "Training Loss":
            ax.set_title(f'Training loss for $\gamma_2={{{gamma_2}}}$')
        elif acc == "Test mse loss":
            ax.set_title(f'Test MSE-loss for $\gamma_2={{{gamma_2}}}$')
        elif acc == "Test_rel_l2_loss":
            ax.set_title(f'Relative $L^2$-loss for $\gamma_2={{{gamma_2}}}$')
        else:
            raise ValueError("This metric is not available!")
        plt.legend(title='$\gamma_1$', loc='lower center', bbox_to_anchor = [0.5, -0.3], ncols = len(gamma_1_list))
        plt.xlabel('Number of Epochs')
        plt.ylabel("Loss")
        if pde == "Poisson":
            plt.yscale('log')
        plt.grid()

        fname = f'plot_gamma2_{gamma_2}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}'

        fig_path = os.path.join(figures_directory, fname)
        ax.figure.savefig(fig_path + '.jpg', dpi=300, bbox_inches='tight')
        plt.close('all')
    return

if __name__ == '__main__':
    pde = "Burgers"
    # "Training Loss", "Test mse loss", "Test_rel_l2_loss"
    acc = "Test mse loss" 
    gamma_1_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hidden_units_1 = 100
    hidden_units_2 = 100
    epochs = 40000
    directory = os.getcwd()
    optimizer="Adam"
    run_accuracy_plots(
        pde=pde,
        epochs=epochs,
        acc=acc,
        gamma_1_list = gamma_1_list,
        gamma_2_list = gamma_2_list,
        hidden_units_1 = hidden_units_1,
        hidden_units_2 = hidden_units_2,
        directory=directory,
        optimizer=optimizer
    )















