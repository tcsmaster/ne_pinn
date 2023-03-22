from Burgers_process import *
from plots import *
pde = 'Poisson'
acc = "Training Loss"
gamma_1_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gamma_2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#gamma_3_list = [0.5, 0.7, 1.0]
hidden_units_1 = 100
hidden_units_2 = 100
#hidden_units_3 = 100
epochs = 10000
directory = os.getcwd()
'''
for gamma_1 in gamma_1_list:
    for gamma_2 in gamma_2_list:
        for gamma_3 in gamma_3_list:
            process(pde=pde,
                    gamma_1=gamma_1,
                    gamma_2=gamma_2,
                    gamma_3 = gamma_3,
                    hidden_units_1=hidden_units_1,
                    hidden_units_2=hidden_units_2,
                    hidden_units_3=hidden_units_3,
                    directory=directory,
                    epochs=epochs
            )
'''
    # Run run_3layer_accuracy_plots(...) for three-layer plots
    # Run run_2layer_accuracy_plots(...) for two-layer plots
    # Run run_2layer_accuracy_plots_multiple_hidden_units(...) to compare different hidden units combination for two-layer neural network
run_2layer_accuracy_plots(
        pde=pde,
        epochs=epochs,
        acc=acc,
        gamma1_list = gamma_1_list,
        gamma2_list = gamma_2_list,
        hidden_units_1 = hidden_units_1,
        hidden_units_2 = hidden_units_2,
        directory=directory)
