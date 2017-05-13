import gym

class Environment(object):
    """
    A wrapper for gym environments.
    """


    def __init__(self, config):
        self.env = gym.make(config.env_name).env
        self.config = config


    def perform(self, action):
        """
        Performs 'action' on the wrapped environment.
        :param action: The action to perform.
        :return: observation, reward, done, info
        """
        return self.env.step(action)


    def render(self):
        self.env.render()


    def reset(self):
        return self.env.reset()


    def reward(self, score):
        """
        Returns a generalizable score translated from the goals for this specific environment.
        A higher reward represents a better performance by the agent.

        Motivation: Since different environments have different goals, raw scores can't be compared
                    between environments. This method computes a value describing the success in this
                    environment.

        :param score: The raw reward received by the environment.
        :return: A score describing the success of an agent in the environment.
        """
        return score
