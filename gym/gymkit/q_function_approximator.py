class DeepQNetwork(object):

    def __init__(self):
        self.critic_age = 0
        self.q_history = []
        self.critic = None
        self.actor = None


    def activate(self, input):
        """
        Feeds the specified `input` vector to the actor network and returns its predicted q vector.
        :param input: A state retrieved from the environment.
        :return: A vector of scalar q values for every action.
        """
        raise NotImplementedError


    def replay(self, experiences):
        """
        Performs experience replay with the given batch of experiences,
        randomly sampled from all experiences stored in the memory.
        :param experiences: A random sample of experiences.
        """
        raise NotImplementedError


    def prepare_for_episode(self):
        """
        Informs the network that a new episode will be run.
        """
        pass


    def did_finish_episode(self, memory):
        """
        Informs the network that an episode was finished.
        :param memory: The current memory. 
        """
        pass
