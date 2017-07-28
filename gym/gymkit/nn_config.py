import numpy as np
from gym.spaces import Box


class FFNNConfig(object):

    def __init__(self, env):
        self.env = env


    @property
    def input_neurons(self):
        """
        The number of input neurons.
        """
        return np.product(self.env.observation_space.shape)


    @property
    def output_neurons(self):
        """
        The number of output neurons.
        """
        if isinstance(self.env.action_space, Box):
            return np.product(self.env.action_space.shape)
        else:
            return self.env.action_space.n


    @property
    def hidden_layer_size(self):
        """
        The number of neurons in the hidden layer.
        """
        return abs(self.input_neurons - self.output_neurons) / 2


    @property
    def input_layer_activation(self):
        """
        The activation function used for input layer. 
        """
        return 'relu'


    @property
    def hidden_layer_activation(self):
        """
        The activation function used for hidden layers. 
        """
        return 'relu'


    @property
    def output_layer_activation(self):
        """
        The activation function used for output layer.
        """
        if isinstance(self.env.action_space, Box):
            # A box action space models a multidimensional regression objective.
            return 'softmax'
        else:
            # A discrete action space models a classification objective.
            return 'sigmoid'


    @property
    def loss_function(self):
        """
        The loss function that the optimizer tries to minimize. 
        """
        return 'binary_crossentropy'


    @property
    def optimizer(self):
        """
        The training algorithm.
        """
        return 'sgd'  # statistical gradient descent


    @property
    def metrics(self):
        return ['accuracy']