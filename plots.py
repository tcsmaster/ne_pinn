import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import generate_file_name

plt.rcParams.update({                      # setup matplotlib to use latex for output
    "font.monospace": [],
    "figure.figsize": (12,8),
    "axes.labelsize": 20,               # LaTeX default is 10pt font.
    "font.size": 20,
    "legend.fontsize": 20,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    })


def load_accuracy_for_single_gamma(pde:str,
                                   epochs:int,
                                   hidden_units_1:int,
                                   hidden_units_2:int,
                                   gamma_1:float,
                                   gamma_2:float,
                                   directory:str,
                                   hidden_units_3=None,
                                   gamma_3 = None,
                                   sampler=None
    ):

    if not sampler:
        if not gamma_3:
            fname = generate_file_name(pde=pde,
                                   epochs=epochs,
                                   hidden_units_1=hidden_units_1,
                                   hidden_units_2=hidden_units_2,
                                   gamma_1=gamma_1,
                                   gamma_2=gamma_2
        )
            results_folder = f'results/{pde}/2layer/normalized/'
        else:
            fname = generate_file_name(pde=pde,
                                   epochs=epochs,
                                   hidden_units_1=hidden_units_1,
                                   hidden_units_2=hidden_units_2,
                                   gamma_1=gamma_1,
                                   gamma_2=gamma_2,
                                   hidden_units_3=hidden_units_3,
                                   gamma_3=gamma_3)
            results_folder = f'results/{pde}/3layer/normalized/'
    else:
        if not gamma_3:
            fname = generate_file_name(pde=pde,
                                   epochs=epochs,
                                   hidden_units_1=hidden_units_1,
                                   hidden_units_2=hidden_units_2,
                                   gamma_1=gamma_1,
                                   gamma_2=gamma_2,
                                   sampler=sampler
        )
            results_folder = f'results/{pde}/2layer/{sampler}/'
        else:
            fname = generate_file_name(pde=pde,
                                   epochs=epochs,
                                   hidden_units_1=hidden_units_1,
                                   hidden_units_2=hidden_units_2,
                                   hidden_units_3 = hidden_units_3,
                                   gamma_1=gamma_1,
                                   gamma_2=gamma_2,
                                   gamma_3 = gamma_3,
                                   sampler=sampler
        )
            results_folder = f'results/{pde}/3layer/{sampler}/'
    # Create full path to data file, including extension
    path = os.path.join(directory, results_folder, fname) + '.csv'
    
    # Load data file
    data = pd.read_csv(path, index_col=0)

    return data


def load_all_accuracy(pde,
                      epochs,
                      acc,
                      gamma_1_list,
                      gamma_2_list, 
                      hidden_units_1,
                      hidden_units_2,
                      directory,
                      hidden_units_3 =None, 
                      gamma_3_list =None
    ):
    """Returns a DataFrame with either test or train accuracy by epoch for  
    lists of gamma values
    
    Parameters
    ----------
    gamma_1_list: list of floats
        the mean-field scaling parameters for the first layer
    gamma_2_list: list of floats
        the mean-field scaling parameters for the second layer
    gamma_3_list: list of floats
        the mean-field scaling parameters for the third layer
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    hidden_units_3: int
        the number of nodes in the third hidden layer
    """

    # Dictionary to store data by different gamma_1
    dict_data = dict()

    # Iterate over list of gamma values and load accuracy data
    if not gamma_3_list:
        for gamma_1 in gamma_1_list:
            for gamma_2 in gamma_2_list:
                data = load_accuracy_for_single_gamma(pde=pde,
                                                      epochs=epochs,
                                                      hidden_units_1=hidden_units_1,
                                                      hidden_units_2=hidden_units_2,
                                                      gamma_1=gamma_1,
                                                      gamma_2=gamma_2,
                                                      directory=directory
                )
                dict_data[(gamma_1,gamma_2)] = data[acc]
    else:
        for gamma_1 in gamma_1_list:
            for gamma_2 in gamma_2_list:
                for gamma_3 in gamma_3_list:
                    data = load_accuracy_for_single_gamma(pde=pde,
                                                          epochs=epochs,
                                                          hidden_units_1=hidden_units_1,
                                                          hidden_units_2=hidden_units_2,
                                                          hidden_units_3=hidden_units_3,
                                                          gamma_1=gamma_1,
                                                          gamma_2=gamma_2,
                                                          gamma_3=gamma_3,
                                                          directory=directory
                    )
                    dict_data[(gamma_1, gamma_2, gamma_3)] = data[acc]
        
    # Concatenate accuracy data over gamma values
    results = pd.concat(dict_data, axis=1)
    return results


def run_2layer_accuracy_plots(pde,
                              epochs,
                              acc,
                              gamma_1_list,
                              gamma_2_list,
                              hidden_units_1,
                              hidden_units_2,
                              directory
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

    # Load accuracy data
    data = load_all_accuracy(pde=pde,
                             epochs=epochs,
                             acc=acc,
                             gamma_1_list=gamma_1_list,
                             gamma_2_list=gamma_2_list,
                             hidden_units_1=hidden_units_1,
                             hidden_units_2=hidden_units_2,
                             directory=directory)
    figures_directory = os.path.join(directory, f"figures/{pde}/2layer/")
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory)
    for gamma_1 in gamma_1_list:
        fig = plt.figure(figsize=(20, 10))
        ax = data[gamma_1].plot()
        ax.set_title(f'{acc} for $\gamma_1={{{gamma_1}}}$ for {pde} pde')
        plt.legend(title='$\gamma_2$', loc='best')
        plt.xlabel('Number of Epochs')
        plt.ylabel(acc)
        plt.yscale('log')
        plt.grid()

        fname = f'plot_{pde}_acc_{acc}_gamma1_{gamma_1}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}'

        fig_path = os.path.join(figures_directory, fname)       
        ax.figure.savefig(fig_path + '.jpg', dpi=300, bbox_inches='tight')
        plt.close('all')      
        
    for gamma_2 in gamma_2_list:
        fig = plt.figure(figsize=(20, 10))
        ax = data.xs(gamma_2, level=1, axis=1).plot()
        ax.set_title(f'{acc} for $\gamma_2={{{gamma_2}}}$')
        plt.legend(title='$\gamma_1$', loc='best')
        plt.xlabel('Number of Epochs')
        plt.ylabel(acc)
        plt.yscale('log')
        plt.grid()

        fname = f'plot_{pde}_acc_{acc}_gamma2_{gamma_2}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}'

        fig_path = os.path.join(figures_directory, fname)
        ax.figure.savefig(fig_path + '.jpg', dpi=300, bbox_inches='tight')
        plt.close('all')
    return

def run_3layer_accuracy_plots(pde,
                              epochs,
                              acc,
                              gamma_1_list,
                              gamma_2_list,
                              gamma_3_list,
                              hidden_units_1,
                              hidden_units_2,
                              hidden_units_3, 
                              directory
    ):
    """Plots and saves figures of test or train accuracy for lists of multiple 
    gamma values for Multi-layer perceptron with three hidden layers (MLP3)
    
    Parameters
    ----------
    gamma_1_list: list of floats
        the mean-field scaling parameters for the first layer 
    gamma_2_list: list of floats
        the mean-field scaling parameters for the second layer 
    gamma_3_list: list of floats
        the mean-field scaling parameters for the third layer 
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    hidden_units_3: int
        the number of nodes in the third hidden layer
    directory: str
        location of the data files and figures
    """

    # Load accuracy data
    data = load_all_accuracy(pde=pde,
                             epochs=epochs,
                             acc=acc,
                             gamma_1_list=gamma_1_list,
                             gamma_2_list=gamma_2_list,
                             gamma_3_list=gamma_3_list,
                             hidden_units_1=hidden_units_1,
                             hidden_units_2=hidden_units_2,
                             hidden_units_3=hidden_units_3,
                             directory=directory)
    
    line_styles = ['solid', 'dashed', 'dotted']
    colors = ['blue', 'green', 'red']
    figures_directory = os.path.join(directory, f'figures/{pde}/3layer')
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory)
    for gamma_1 in gamma_1_list:
        
        fig = plt.figure(figsize=(20, 10))
        
        legend_labels = []
        for count_1, gamma_2 in enumerate(gamma_2_list):
            for count_2, gamma_3 in enumerate(gamma_3_list):
                legend_labels += [(gamma_2, gamma_3)]
                plt.plot(data[(gamma_1, gamma_2, gamma_3)],
                         color=colors[count_1],
                         linestyle=line_styles[count_2])
      
        ax = fig.axes[0]
        plt.ylabel(acc)
        ax.set_title(f'{acc} for $\gamma_1={{{gamma_1}}}$ for {pde} pde')
        plt.legend(legend_labels, title='$(\gamma_2, \gamma_3)$', loc='best', ncol=3)
        plt.xlabel('Number of Epochs')
        plt.yscale('log')
        plt.grid()

        fname = f'plot_{pde}_acc_{acc}_gamma1_{gamma_1}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_hidden3_{hidden_units_3}'
    
        fig_path = os.path.join(figures_directory, fname)
        ax.figure.savefig(fig_path + '.jpg', dpi=300, bbox_inches='tight')
        plt.close('all')
    for gamma_2 in gamma_2_list:
        
        fig = plt.figure(figsize=(20, 10))
        
        legend_labels = []
        for count_1, gamma_1 in enumerate(gamma_1_list):

            for count_2, gamma_3 in enumerate(gamma_3_list):
                legend_labels += [(gamma_1, gamma_3)]
                plt.plot(data[(gamma_1, gamma_2, gamma_3)],
                         color=colors[count_1],
                         linestyle=line_styles[count_2])

        ax = fig.axes[0]
        plt.ylabel(acc)
        ax.set_title(f'{acc} for $\gamma_2={{{gamma_2}}}$ for {pde} pde')
        plt.legend(legend_labels, title='$(\gamma_1, \gamma_3)$', loc='best', ncol=3)
        plt.xlabel('Number of Epochs')
        plt.yscale('log')
        plt.grid()
    
        fname = f'plot_{pde}_acc_{acc}_gamma2_{gamma_2}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_hidden3_{hidden_units_3}'
    
        fig_path = os.path.join(figures_directory, fname)
        ax.figure.savefig(fig_path + '.jpg', dpi=300, bbox_inches='tight')
        plt.close('all')
    for gamma_3 in gamma_3_list:
        
        fig = plt.figure(figsize=(20, 10))
        
        legend_labels = []
        for count_1, gamma_1 in enumerate(gamma_1_list):
            for count_2, gamma_2 in enumerate(gamma_2_list):
                legend_labels += [(gamma_1, gamma_2)]
                plt.plot(
                    data[(gamma_1, gamma_2, gamma_3)],
                    color=colors[count_1],
                    linestyle=line_styles[count_2])

        ax = fig.axes[0]
        plt.ylabel(acc)
        ax.set_title(f'{acc} for $\gamma_3={{{gamma_3}}}$ for {pde} pde')
        plt.legend(legend_labels, title='$(gamma_1, gamma_2)$', loc='best', ncol=3)
        plt.xlabel('Number of Epochs')
        plt.yscale('log')
        plt.grid()

        fname = f'plot_{pde}_acc_{acc}_gamma3_{gamma_3}_hidden1_{hidden_units_1}_hidden2_{hidden_units_2}_hidden3_{hidden_units_3}'

        fig_path = os.path.join(figures_directory, fname)
        ax.figure.savefig(fig_path + '.jpg', dpi=300, bbox_inches='tight')
        plt.close('all')
    return


def run_2layer_accuracy_plots_multiple_hidden_units(pde,
                                                    epochs,
                                                    acc,
                                                    gamma_1,
                                                    gamma_2,
                                                    hidden_units_list_1,
                                                    hidden_units_list_2,
                                                    directory
    ):
    """
    Plots and saves figures of test or train accuracy which compare 
    different pairs of hidden units for fixed gammas for MLP2.    
    
    Parameters
    ---------- 
    model_name: str
        'mlp2' 
    directory: str
        location of the data files and figures
    """
    
    dict_data = dict()
    
    for hidden_units_1, hidden_units_2 in zip(hidden_units_list_1, hidden_units_list_2):
    
        dict_data[(hidden_units_1, hidden_units_2)] = load_accuracy_for_single_gamma(
            pde=pde,
            epochs=epochs,
            gamma_1=gamma_1, 
            gamma_2=gamma_2, 
            hidden_units_1=hidden_units_1, 
            hidden_units_2=hidden_units_2, 
            directory=directory)[acc]
                
    data = pd.concat({
        f'N1={hidden_units[0]},N2={hidden_units[1]}': dict_data[hidden_units]
        for hidden_units in dict_data.keys()
        }, axis=1)
    
    # Create a new figure and plot accuracy data
    fig = plt.figure(figsize=(20, 10))
    ax = data.plot()

    # Set title, label legend and x- and y-axes
    ax.set_title(f'{acc}:  gamma_1={gamma_1}, gamma_2={gamma_2} for {pde} pde')
    plt.legend(title='hidden units', loc='center left', bbox_to_anchor=(1,0.5))
    plt.xlabel('Number of Epochs')
    y_label = acc
    plt.ylabel(y_label)
    plt.yscale('log')
   
    # Generate file name
    fname = f'plot_{pde}_acc_{acc}_gamma1_{gamma_1}_gamma2_{gamma_2}'
    
    # Generate full path
    figures_directory = os.path.join(directory, f'figures/{pde}/')
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory)
    fig_path = os.path.join(figures_directory, fname)
    
    
    ax.figure.savefig(fig_path + '.jpg', dpi=300, bbox_inches='tight')
    #ax.figure.savefig(fig_path + '.pdf', dpi=300, bbox_inches='tight')
    plt.close('all')
    return

if __name__ == '__main__':
    pde = "Burgers"
    acc = "Training Loss"
    gamma_1_list = [0.5, 0.7, 1.0]
    gamma_2_list = [0.5, 0.7, 1.0]
    gamma_3_list = [0.5, 0.7, 1.0]
    hidden_units_1 = 100
    hidden_units_2 = 100
    hidden_units_3 = 100
    epochs = 20000
    directory = os.getcwd()

    run_3layer_accuracy_plots(pde=pde,
                              epochs=epochs,
                              acc=acc,
                              gamma_1_list = gamma_1_list,
                              gamma_2_list = gamma_2_list,
                              gamma_3_list=gamma_3_list,
                              hidden_units_1 = hidden_units_1,
                              hidden_units_2 = hidden_units_2,
                              hidden_units_3 = hidden_units_3,
                              directory=directory)















