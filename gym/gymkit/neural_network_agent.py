from keras.models import Sequential
from keras.layers import Dense
from gymkit.agent import Agent
from gym.spaces import Box
import numpy as np
from gymkit.nn_config import FFNNConfig


class NeuralNetworkAgent(Agent):

    def __init__(self, id='NeuralNetworkAgent', hidden_layers=1, quality_score_threshold=50, verbose=False):
        super(NeuralNetworkAgent, self).__init__(id)
        self.env = None
        self.config = FFNNConfig(None)
        self.hidden_layers = hidden_layers
        self.quality_score_threshold = quality_score_threshold
        self.model = None
        self.episode_count = 0
        self.gradient_updates = 0
        self.quality_episodes = []


    def build_model(self):
        """
        Builds and returns a neural network.
        """
        # model = Sequential()
        # model.add(Dense(16, activation=self.input_layer_activation, input_shape=(self.input_neurons,)))
        # model.add(Dense(self.hidden_layer_size, activation=self.hidden_layer_activation))
        # model.add(Dense(self.output_neurons, activation=self.output_layer_activation))
        # model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        model = Sequential()
        model.add(Dense(self.config.hidden_layer_size, activation='relu', input_dim=self.config.input_neurons))
        model.add(Dense(self.config.output_neurons, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        return model


    def setup(self, environment):
        self.env = environment
        self.config = FFNNConfig(environment)
        self.model = self.build_model()


    def action(self, prediction):
        """
        Returns an action that the model predicts minimizes
        the loss after having experienced `observation`.
        """
        # Returns a random sample from the action space if the network was not trained yet.
        if self.gradient_updates <= 50:
            return self.env.action_space.sample()

        if isinstance(self.env.action_space, Box):
            return prediction.reshape(self.env.action_space.shape)
        else:
            return np.argmax(prediction)


    def activate(self, observation):
        """
        Activates the neural network by feeding an observation vector to the input layer.
        :param observation: The observation that the input vector is constructed from.
        :return: The values of the activated output neurons that the model predicts minimize
                 the loss after having experienced `observation`.
        """
        input = np.array(observation).reshape((1, 4))
        return self.model.predict(input).reshape(self.config.output_neurons)


    def train(self, dataset, epochs=10):
        input_vector = np.array(zip(*dataset)[0])
        output_vector = np.array(zip(*dataset)[1])
        self.model.fit(input_vector, output_vector, epochs)
        self.gradient_updates += 1
        return self.model


    def log_quality_episode(self, samples):
        self.quality_episodes.append(samples)
        if len(self.quality_episodes) > 100:
            quality_scores = np.array(self.quality_episodes)
            samples = quality_scores.reshape(np.product(quality_scores.shape))
            self.train(samples)


    def run_episode(self, render=False):
        self.episode_count += 1
        episode_reward = 0
        episode_dataset = []
        observation = self.env.reset()

        while True:
            prediction = self.activate(observation)
            episode_dataset.append((observation, prediction))
            action = self.action(prediction)
            observation, reward, done, _ = self.env.perform(action)
            episode_reward += reward

            if render:
                self.env.render()

            if done:
                if episode_reward >= self.quality_score_threshold:
                    self.log_quality_episode(episode_dataset)
                return episode_reward


