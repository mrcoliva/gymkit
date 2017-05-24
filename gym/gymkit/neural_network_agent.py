from keras.models import Sequential
from keras.layers import Dense
from gymkit.agent import Agent
from gym.spaces import Box
import numpy as np


class NeuralNetworkAgent(Agent):

    def __init__(self, id='NeuralNetworkAgent', hidden_layers=1, verbose=False):
        super(NeuralNetworkAgent, self).__init__(id)
        self.env = None
        self.hidden_layers = hidden_layers
        self.model = None


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
            return 'softmax'


    @property
    def loss_function(self):
        """
        The loss function that the optimizer tries to minimize. 
        """
        return 'binary_crossentropy'


    @property
    def optimzer(self):
        """
        The training algorithm.
        """
        return 'sgd'  # statistical gradient descent


    @property
    def metrics(self):
        return ['accuracy']


    def build_model(self):
        """
        Builds and returns a neural network.
        """
        model = Sequential()
        model.add(Dense(self.input_neurons, activation=self.input_layer_activation, input_dim=self.input_neurons))
        model.add(Dense(self.hidden_layer_size, activation=self.hidden_layer_activation, input_dim=self.hidden_layer_size))
        model.add(Dense(self.output_neurons, activation=self.output_layer_activation))
        model.compile(optimizer=self.optimzer, loss=self.loss_function, metrics=self.metrics)
        return model


    def setup(self, environment):
        self.env = environment
        self.model = self.build_model()


    def action(self, observation):
        # prediction = self.model.predict(np.array(observation))
        # if isinstance(self.env.action_space, Box):
        #     return prediction.reshape(self.env.action_space.shape)
        # else:
        #     return np.argmax(prediction)
        return self.env.action_space.sample()


    def run_episode(self, render=False):
        episode_reward = 0
        observation = self.env.reset()

        while True:
            observation, reward, done, _ = self.env.perform(self.action(observation))
            episode_reward += reward

            if render:
                self.env.render()

            if done:
                return episode_reward


