import gym
from gym.spaces import Box
import numpy as np


class Environment(object):
    """
    A wrapper for gym environments.
    """

    def __init__(self, name, mode='training', monitoring_enabled=False):
        self.name = name
        self.mode = mode
        self.env = gym.make(self.name)
        # if monitoring_enabled:
        #     self.env = wrappers.Monitor(self.env, self.outdir(), force=True, mode=self.mode)

        self.episode_count = 0


    @property
    def outdir(self):
        return '/tmp/{}'.format(self.name)


    @property
    def reward_threshold(self):
        """
        :return: The reward threshold before the task is considered solved. 
        """
        return self.env.spec.reward_threshold


    @property
    def has_discrete_action_space(self):
        return not isinstance(self.env.action_space, Box)


    @property
    def trials(self):
        """
        :return: The number of trials in which the average reward goal must be reached to solve. 
        """
        return self.env.spec.trials


    @property
    def max_episode_steps(self):
        """
        :return: The maximum number of steps before an episode is reset.
        """
        return self.env.spec.max_episode_steps


    @property
    def observation_space(self):
        return self.env.observation_space


    @property
    def action_space(self):
        return self.env.action_space


    @property
    def state_vector_length(self):
        if isinstance(self.env.observation_space, Box):
            return self.env.observation_space.shape[0]
        return self.env.observation_space.n


    def perform(self, action):
        """
        Performs 'action' on the wrapped environment.
        :param action: The action to perform.
        :return: observation, reward, done, info
        """
        self.episode_count += 1
        o, r, d, i, = self.env.step(action)
        return np.asarray(o).reshape(1, self.state_vector_length), r, d, i


    def solved(self, scores):
        return False


    def render(self):
        """
        Renders the current state of the environment.
        """
        self.env.render()


    def reset(self):
        """
        Resets the environment to the initial state.
        """
        # return np.asarray(self.env.reset()).reshape(1, self.state_vector_length)
        return self.env.reset()


    def reward(self, score):
        """
        Returns a generalizable score translated from the goals for this specific environment.
        The absolute magnitude and signs of the reward are not important, only their relative values
        where a higher value represents a better performance by the agent.
        Genomes of evolutionary algorithms are advised to use this function to compute their fitness value.

        Motivation: Since different environments have different goals, raw scores can't be compared
                    between environments. This method computes a value describing the success in this
                    environment.

        :param score: The raw reward received by the environment.
        :return: A score describing the success of an agent in the environment.
        """
        return score
