from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from gymkit.q_function_approximator import DeepQNetwork
from keras import activations, losses


class FeedForwardNeuralNetwork(DeepQNetwork):

    def __init__(self, input_neurons, output_neurons, discount=0.9, max_critic_age=300):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.max_critic_age = max_critic_age
        self.discount = discount
        self.actor = self.build_model()
        self.critic = self.build_model()


    def build_model(self):
        """
        Builds and returns a fully-connected feed-forward neural network.
        """
        model = Sequential()
        model.add(Dense(16, activation=activations.relu, input_dim=self.input_neurons))
        model.add(Dense(self.output_neurons, activation=activations.linear))
        model.compile(optimizer='sgd', loss=losses.mse)
        return model


    def activate(self, input):
        """
        Feeds the specified `input` vector to the actor network and returns its predicted q vector.
        :param input: A state retrieved from the environment.
        :return: A vector of scalar q values for every action.
        """
        return self.actor.predict(input)


    def update_critic(self):
        """
        Updates the critic model depending on the elapsed time steps since the last weight replacement.
        """
        if self.critic_age <= self.max_critic_age:
            self.critic_age += 1
        else:
            "Updating critic..."
            self.critic.set_weights(self.actor.get_weights())
            self.critic_age = 0


    def replay(self, experiences):
        """
        Performs experience replay with the given batch of experiences,
        randomly sampled from all experiences stored in the memory.
        :param experiences: A random sample of experiences.
        """
        for experience in experiences:
            q_state = self.activate(experience.state)
            q_result_state = self.critic.predict(experience.result_state)
            target = q_state.copy()

            if experience.terminal:
                target[0, experience.action] = experience.reward
            else:
                max_future_q = np.amax(q_result_state[0])
                target[0, experience.action] = experience.reward + self.discount * max_future_q
                self.q_history.append(max_future_q)

            self.actor.train_on_batch(experience.state, target)
            self.update_critic()
