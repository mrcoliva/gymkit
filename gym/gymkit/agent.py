class Agent(object):

    def __init__(self, id='UnnamedAgent'):
        self.id = id

    @staticmethod
    def epsilon(t, max=1, min=0.02, max_t=10000):
        """
        Returns an epsilon for the epsilon-greedy policy at time step t, linearly annealed 
        between from max to min over max_t time steps, and fixed at min afterwards.

        :param min: The minimum epsilon, used as the fixed value after max_t time steps.
        :param max: The maximum epsilon, used as the initial value.
        :param max_t: The number of time steps over which epsilon is linearly annealed from max to min.
        :return: The scalar value of epsilon.
        """
        if t >= max_t:
            return min
        return max - (t * (max - min) / max_t)


    def setup(self, environment):
        raise NotImplementedError


    def run_episode(self, render=False):
        raise NotImplementedError


    def evaluate(self, max_episodes):
        pass
