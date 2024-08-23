import torch.nn as nn
import torch

class MultilayerPerceptron(nn.Module):
    def chooseActivation(self, act_str):
        if act_str == 'relu':
            return nn.ReLU()
        elif act_str == 'sigmoid':
            return nn.Sigmoid()
        elif act_str == 'tanh':
            return nn.Tanh()
        elif act_str == 'linear':
            return nn.Identity()
        else:
            return nn.ReLU()


    def __init__(self, input_size, number_hidden_layers, cells_per_hidden_layer, 
                                 output_layer_size, batch_norm=False, dropout=False,
                                activation_hidden='relu',
                                activation_output='linear'):
        """
        Creates a classic multilayer perceptron with the specified parameters. 
        It can add batch normalization and dropout.
        
        :param input_size: Size of the input layer
        :param number_hidden_layers: Number of hidden layers
        :param cells_per_hidden_layer: Array of the number of neurons in each hidden layer
        :param output_layer_size: Size of the output layer
        :param batch_norm: Boolean to use batch normalization after each hidden layer
        :param dropout: Float value for dropout probability (0 < dropout <= 1). Set False to disable.
        :param activation_hidden: Activation function for hidden layers ('relu', 'tanh', etc.)
        :param activation_output: Activation function for output layer ('sigmoid', 'softmax', etc.)
        """
    
        super(MultilayerPerceptron, self).__init__()
        
        self.layers = nn.ModuleList()  # Use ModuleList to store the layers
        
        current_input_size = input_size

        self.activation_hidden = self.chooseActivation(activation_hidden)
        self.batch_norm = batch_norm
        
        # Hidden layers
        for cur_layer in range(number_hidden_layers):
            self.layers.append(nn.Linear(current_input_size, cells_per_hidden_layer[cur_layer]))
            
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(cells_per_hidden_layer[cur_layer]))

            self.layers.append(self.chooseActivation(self.activation_hidden))
            # if dropout:
                # self.layers.append(nn.Dropout(p=dropout))
            
            current_input_size = cells_per_hidden_layer[cur_layer]
        
        # Output layer
        self.output_layer = nn.Linear(current_input_size, output_layer_size)
        self.activation_output = self.chooseActivation(activation_output)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        x = self.activation_output(x)
        
        return x