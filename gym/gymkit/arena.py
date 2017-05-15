from gymkit.agent import Agent
from collections import defaultdict


class Arena(object):

    def __init__(self, environment_class, mode='training', monitoring_enabled=True):
        """
        Initializes a new `Arena`.
        :param environment_class: The class of the environment that should be evaluated.
        """
        self.Environment = environment_class
        self.mode = mode
        self.monitoring_enabled = monitoring_enabled
        self.agents = []
        self.episodes = defaultdict(lambda: [])
        self.environments = {}


    def reset(self):
        """
        Resets all saved states and scores.
        Note: Registered agents won't be removed.
        """
        self.episodes = {}


    def register(self, agent):
        """
        Registers an agent and configures it with an environment.
        :param agent: An agent which can perform episodes on random environments and 
        be evaluated based on success.
        """
        assert isinstance(agent, Agent), 'Tried to register an invalid agent. Agents must be of class `Agent`.'
        assert agent.id not in self.environments, 'Agent with id: "{}" already registered.'.format(agent.id)

        agent.setup(self.environment(agent))
        self.agents.append(agent)


    def environment(self, agent):
        """
        Returns the environment that the 'agent' performs in. If no environment was created
        for the 'agent', than a new one is initialized with the current configuration.
        :param agent: The agent whose environment is requested.
        :return: An environment.
        """
        if agent.id not in self.environments:
            self.environments[agent.id] = self.Environment(mode='training', monitoring_enabled=True)

        return self.environments[agent.id]


    def log_episode(self, agent, score):
        """
        Stores the score of an episode for the 'agent'.
        :param agent: The action which played the episode.
        :param score: The score reached in the episode.
        """
        self.episodes[agent.id].append(score)
        print 'Episode {}, Score: {}'.format(len(self.episodes[agent.id]), score)


    def process_finished_episode(self, agent, score):
        self.log_episode(agent, score)


    def num_episodes_of(self, agent):
        """
        Returns the number of episodes finished by the 'agent'.
        :param agent: The agent whose number of finished episodes is requested.
        :return: The number of episodes finished by the 'agent'.
        """
        return len(self.episodes[agent.id])


    def run(self, max_episodes=100, render=False):
        """
        Runs each registered agent on the environment until the environment is solved or the specified
        number of episodes is reached.
        :return: A dictionary containing an evaluation for each agent.
        """
        evaluations = {}

        for agent in self.agents:
            env = self.environment(agent)
            while self.num_episodes_of(agent) < max_episodes:
                score = agent.run_episode(env)
                self.process_finished_episode(agent, score)

        return evaluations
