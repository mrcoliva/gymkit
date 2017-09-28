import neat, os, numpy as np
import time
from neat.nn import FeedForwardNetwork
from gymkit.agent import Agent
from gymkit.environment import Environment
from gymkit.evaluation import Evaluation


def variance(values: [float]):
    mean = np.mean(values)
    values = list(map(lambda x: np.square(x - mean), values))
    return np.sum(values) / len(values)


class NeatAgent(Agent):

    def __init__(self, id='NeatAgent', elite_size=3, test_episodes=1, verbose=False):
        super(NeatAgent, self).__init__(id)
        self.env = None
        self.config = None
        self.verbose = verbose
        self.stats = neat.StatisticsReporter()
        self.evaluator = neat.ParallelEvaluator(2, self.compute_fitness)
        self.generation = 0
        self.population = None
        self.elite_size = elite_size
        self.test_episodes = test_episodes
        self.scores = []
        self.fittest_genome = None
        self.t0 = None


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
        print('{0}: Score: {1} | {2} ({3})'.format(len(self.scores), self.scores[-1], self.stats.best_unique_genomes(1)[0].fitness, np.mean(self.scores[-99:])))


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


    def average_score(self, network: neat.nn.FeedForwardNetwork):
        """
        Runs the network in an environment and measures its success. 
        :param network: The network to evaluate.
        :return: The average score reached in the test episodes.
        """
        scores = []

        for _ in range(self.test_episodes):
            state = self.env.reset()
            e_score = 0

            while True:
                observation, reward, done, _ = self.env.perform(self.action(state, [network]))
                state = observation
                e_score += reward

                if done:
                    scores.append(e_score)
                    break

        return np.mean(scores)


    def evolve(self, generations: int = 1):
        """
        Runs the genetic algorithm to evolve the population for the specified number of generations.
        """
        self.population.run(fitness_function=self.compute_fitness, n=generations)
        self.generation += generations
        self.update_fittest_genome_if_needed()
        print('Elite: {}'.format(list(map(lambda g: (g.key, g.fitness), self.stats.best_unique_genomes(self.elite_size)))))


    def update_fittest_genome_if_needed(self):
        fittest = self.stats.best_genome()
        if self.fittest_genome is None:
            print('[{0}] Setting initial fittest genome ({1})'.format(self.id, int(fittest.fitness)))
            self.fittest_genome = fittest
        if fittest.fitness > self.fittest_genome.fitness:
            print('[{0}] Updated fittest genome ({1} -> {2}) in gen. {3}.'
                  .format(self.id, int(self.fittest_genome.fitness), int(fittest.fitness), self.generation))
            self.fittest_genome = fittest


    def action(self, state: np.ndarray, networks: [neat.nn.FeedForwardNetwork]):
        votes = [network.activate(state.flatten()) for network in networks]
        aggregated_votes = list(map(self.aggregate_output, list(zip(*votes))))

        if self.env.has_discrete_action_space:
            return np.argmax(aggregated_votes)
        else:
            return aggregated_votes


    @staticmethod
    def aggregate_output(output: np.ndarray) -> np.ndarray:
        """
        The function used to aggregate the output of multiple phenotypes into a single decision. 
        """
        return np.mean(output)


    @property
    def actors(self) -> [FeedForwardNetwork]:
        """
        Returns the agent's actor networks.
        """
        return self.fittest_networks(self.elite_size)


    def evaluation(self) -> Evaluation:
        return Evaluation(name=self.id, info={
            "scores": self.scores,
            'runtime': time.time() - self.t0,
            'best_fitness': [c.fitness for c in self.stats.most_fit_genomes],
            'avg_fitness': self.stats.get_fitness_mean(),
            'stdev_fitness': self.stats.get_fitness_stdev(),
            'species_sites': self.stats.get_species_sizes(),
            'best_genome': str(self.stats.best_genome())
        })


    def evaluate(self, max_episodes: int, render=False) -> [float]:
        self.t0 = time.time()
        self.scores = []
        # env = self.env
        env = Environment(self.env.name)  # a temporal workaround for a bug that caused the env to be already done

        while len(self.scores) < max_episodes and not env.solved(self.scores):
            state = env.reset()
            episode_reward = 0
            self.evolve(generations=1)
            actors = self.actors

            while True:
                action = self.action(state, actors)
                observation, reward, done, _ = env.perform(action)
                state = observation
                episode_reward += reward

                if render:
                    env.render()

                if done:
                    self.log_episode(episode_reward)
                    break

        return self.evaluation()
