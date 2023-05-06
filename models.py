import torch
import torch.nn as nn

class MLP2(nn.Module):
    """
    Multi-layer perceptron with two hidden layers.
    
    Attributes
    ----------
    num_input: int
        The dimension of the network input, the dimension of the PDE
    num_output: int
        The dimension of the network output.
        num_output = 1 => scalar output
        num_output > 1 => vector output
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    gamma_1: float
        the scaling parameter for the first hidden layer
    gamma_2: float
        the scaling parameter for the second hidden layer
    """
    
    def __init__(
        self,
        num_input,
        num_output,
        hidden_units_1,
        hidden_units_2,
        gamma_1,
        gamma_2
    ):
        """
        Initializes the network layers and the network's parameters.
        """
        super(MLP2, self).__init__()
        
        # Parameters
        self.num_input = num_input
        self.num_output = num_output
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        # Layers
        self.fc1 = nn.Linear(self.num_input,self.hidden_units_1)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        self.fc2 = nn.Linear(self.hidden_units_1,self.hidden_units_2)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)
        self.fc3 = nn.Linear(self.hidden_units_2,self.num_output)
        nn.init.uniform_(self.fc3.weight, a=0.0, b=1.0)
    
    def forward(self, x):
        """
        Performs the forward pass of the network.
        """
        scaling_1 = self.hidden_units_1 ** (-self.gamma_1)
        x = scaling_1 * torch.tanh(self.fc1(x))
        scaling_2 = self.hidden_units_2**(-self.gamma_2)
        x = scaling_2 * torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
