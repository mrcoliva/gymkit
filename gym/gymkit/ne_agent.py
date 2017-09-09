import neat, os, time, numpy as np
from neat.nn import FeedForwardNetwork
from gymkit.agent import Agent
from gymkit.environment import Environment
from gym.spaces import Box


class NeatAgent(Agent):

    def __init__(self, id='NeatAgent', elite_size=3, test_episodes=10, verbose=False):
        super(NeatAgent, self).__init__(id)
        self.env = None
        self.config = None
        self.verbose = verbose
        self.stats = neat.StatisticsReporter()
        self.generation = 0
        self.population = None
        self.elite_size = elite_size
        self.test_episodes = test_episodes
        self.scores = []
        self.elite_scores = []


    def setup(self, environment: Environment):
        self.env = environment
        self.config = self.read_config(environment)
        self.population = neat.Population(self.config)
        self.population.add_reporter(self.stats)

        if self.verbose:
            self.population.add_reporter(neat.StdOutReporter(self.config))


    @staticmethod
    def read_config(environment: Environment):
        config_path = os.path.join(os.path.dirname(__file__), 'neat_config/neat-config-{}'.format(environment.name))
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                           neat.DefaultStagnation, config_path)


    def log_episode(self, score: int):
        self.scores.append(score)


    def fittest_networks(self, n: int) -> [neat.nn.FeedForwardNetwork]:
        """
        Returns the fittest n networks of the current population.
        :rtype: list
        :param n: The number of fittest networks to create.
        :return: An array of n neural network phenotypes.
        """
        return [self.phenotype(genome) for genome in self.stats.best_unique_genomes(n)]


    def phenotype(self, genome: neat.DefaultGenome) -> neat.nn.FeedForwardNetwork:
        """
        Creates and returns the neural network phenotype of the given genome. 
        :param genome: The genome encoding the network.
        """
        return FeedForwardNetwork.create(genome, self.config)


    def compute_fitness(self, population: [neat.DefaultGenome], config: neat.Config) -> None:
        """
        Computes the fitness of each genome in the population and 
        stores it in their fitness properties.
        
        :param population: A list of genome_id, genome tuples.
        :param config: The neat config.
        """
        for genome, network in [(genome, self.phenotype(genome)) for _, genome in population]:
            genome.fitness = self.average_score(network)


    def average_score(self, network: neat.nn.FeedForwardNetwork) -> float:
        """
        Runs the network in an environment and measures its success. 
        :param network: The network to evaluate.
        :return: The average score reached in the test episodes.
        """
        total_score = 0
        t = 0

        for _ in range(self.test_episodes):
            state = self.env.reset()

            while True:
                t += 1
                observation, reward, done, _ = self.env.perform(self.action(state, [network], t))
                state = observation
                total_score += reward

                if done:
                    break

        return total_score / self.test_episodes


    def evolve(self, generations: int = 1):
        """
        Runs the genetic algorithm to evolve the population for the specified number of generations.
        """
        self.population.run(fitness_function=self.compute_fitness, n=generations)
        self.generation += generations


    def action(self, state: np.ndarray, networks: [neat.nn.FeedForwardNetwork], t: int):
        # if random.random() < 0.2:  # self.epsilon(t):
        #     return self.env.action_space.sample()

        votes = [network.activate(state.flatten()) for network in networks]
        if isinstance(self.env.action_space, Box):
            return list(map(self.aggregate_output, list(zip(*votes))))
        else:
            return np.argmax(list(map(self.aggregate_output, list(zip(*votes)))))


    @staticmethod
    def aggregate_output(output: np.ndarray) -> np.ndarray:
        """
        The function used to aggregate the output of multiple phenotypes into a single decision. 
        """
        return np.mean(output)


    def actors(self) -> [FeedForwardNetwork]:
        return self.fittest_networks(self.elite_size)


    def evaluate(self, max_episodes: int, render=False) -> [float]:
        self.scores = []
        # env = self.env
        env = Environment(self.env.name)  # a temporal workaround for a bug that caused the env to be already done
        t = 0

        while len(self.scores) < max_episodes:
            state = env.reset()
            episode_reward = 0
            self.evolve(generations=1)

            while True:
                action = self.action(state, self.actors(), t)
                observation, reward, done, _ = env.perform(action)
                state = observation
                episode_reward += reward
                t += 1

                if render:
                    env.render()

                if done:
                    self.log_episode(episode_reward)
                    break

        return self.scores
