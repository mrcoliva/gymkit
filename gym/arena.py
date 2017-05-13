from gym import wrappers
from agent import Agent
from neuro_net import NeatAgent
from collections import defaultdict
from environment import Environment


class CartPoleConfig(object):

    def __init__(self, mode='training', monitor=False):
        self.env_name = 'CartPole-v0'
        self.outdir = '/tmp/{}'.format(self.env_name)
        self.mode = mode
        self.monitor = monitor
        self.min_reward = 0
        self.max_reward = 200
        self.solved_threshhold = 195


    def environment(self):
        env = Environment(self)

        if self.monitor:
            return wrappers.Monitor(env, self.outdir, force=True, mode=self.mode)
        return env


class Arena(object):

    def __init__(self, config):
        self.config = config
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
            self.environments[agent.id] = self.config.environment()

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


    def run(self, max_episodes=10, render=False):
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
                if score >= 1000: break

        return evaluations


if __name__ == '__main__':
    config = CartPoleConfig()
    stadium = Arena(config)
    stadium.register(NeatAgent(elite_size=3))

    evaluations = stadium.run()
    print 'Evaluations: {}'.format(evaluations)
