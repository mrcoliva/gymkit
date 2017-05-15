import gym


class Environment(object):
    """
    A wrapper for gym environments.
    """


    def __init__(self, config):
        self.env = gym.make(config.env_name)
        self.config = config
        self.episode_count = 0


    def perform(self, action):
        """
        Performs 'action' on the wrapped environment.
        :param action: The action to perform.
        :return: observation, reward, done, info
        """
        self.episode_count += 1
        return self.env.step(action)


    def render(self):
        """
        Renders the current state of the environment.
        """
        self.env.render()


    def reset(self):
        """
        Resets the environment to the initial state.
        """
        return self.env.reset()


    def reward(self, score):
        """
        Returns a generalizable score translated from the goals for this specific environment.
        A higher score represents a better performance by the agent.

        Motivation: Since different environments have different goals, raw scores can't be compared
                    between environments. This method computes a value describing the success in this
                    environment.

        :param score: The raw reward received by the environment.
        :return: A score describing the success of an agent in the environment.
        """
        return score


    def reward_threshold(self):
        """
        :return: The reward threshold before the task is considered solved. 
        """
        return self.env.spec.reward_threshold


    def trials(self):
        """
        :return: The number of trials in which the average reward goal must be reached to solve. 
        """
        return self.env.spec.trials


    def max_episode_steps(self):
        """
        :return: The maximum number of steps before an episode is reset.
        """
        return self.env.spec.max_episode_steps
