from Burgers_process import *
from plots import *
pde = "Burgers"
acc = "Training Loss"
gamma_1_list = [0.5, 0.7, 1.0]
gamma_2_list = [0.5, 0.7, 1.0]
gamma_3_list = [0.5, 0.7, 1.0]
hidden_units_1 = 20
hidden_units_2 = 20
hidden_units_3 = 20
epochs = 20000
directory = os.getcwd()

run_3layer_accuracy_plots(
        pde=pde,
        epochs=epochs,
        acc=acc,
        gamma_1_list = gamma_1_list,
        gamma_2_list = gamma_2_list,
        gamma_3_list=gamma_3_list,
        hidden_units_1 = hidden_units_1,
        hidden_units_2 = hidden_units_2,
        hidden_units_3 = hidden_units_3,
        directory=directory)
