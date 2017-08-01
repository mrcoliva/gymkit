from gymkit.q_function_approximator import DeepQNetwork
import neat
import os
import multiprocessing
from neat.nn import FeedForwardNetwork
import numpy as np
import random
import time
from gymkit.q_models import Experience, Memory


def mse(x, y):
    assert len(x) == len(y)
    assert np.asarray(x).shape == np.asarray(y).shape
    return np.mean([(pred - target)**2 for pred, target in zip(x, y)])


class ErrorCalculator(object):

    def __init__(self, env):
        self.env = env
        self.test_episodes = []
        self.generation = 0
        self.simulation_episodes = 20

        self.min_reward = -200
        self.max_reward = 200

        self.episode_score = []
        self.episode_length = []
        self.memory = Memory()


    def simulate(self, nets):
        for genome, net in nets:
            state = self.env.reset()

            while True:
                if random.random() < 0.2:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(net.activate(state.flatten()))

                observation, reward, done, _ = self.env.perform(action)
                self.memory.store(Experience(state, action, reward, observation, done))
                state = observation

                if done:
                    break


    def error(self, genome, net, episodes, min_reward, max_reward):
        states, targets = [], []

        for experience in self.memory.sample(32):
            q_state = net.activate(experience.state.flatten())
            q_result_state = net.activate(experience.result_state.flatten())
            target = np.asarray(q_state).copy()

            if experience.terminal:
                target[experience.action] = experience.reward
            else:
                target[experience.action] = experience.reward + genome.discount * np.amax(q_result_state)

            states.append(q_result_state)
            targets.append(target.flatten())

        return mse(states, targets)


    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        nets = []
        for gid, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))

        # Periodically generate a new set of episodes for comparison.
        t0 = time.time()
        self.test_episodes = self.test_episodes[-20:]
        if self.generation % 10 == 1:
            self.simulate(nets)
            print("Running simulation took {0} seconds.".format(time.time() - t0))

        # Assign a composite fitness to each genome; genomes can make progress either
        # by improving their total reward or by making more accurate reward estimates.
        print("Evaluating {0} test episodes...".format(len(self.test_episodes)))
        t0 = time.time()
        best = -10000
        for genome, net in nets:
            genome.fitness = random.random
            # -self.error(genome, net, self.test_episodes, self.min_reward, self.max_reward)
            if genome.fitness > best:
                best = genome.fitness

        print("Computing fitness took {0} seconds.\n".format(time.time() - t0))
        print('Best fitness is {}'.format(best))


class QGenome(neat.DefaultGenome):

    def __init__(self, key):
        super(QGenome, self).__init__(key)
        self.discount = None


    def configure_new(self, config):
        super(QGenome, self).configure_new(config)
        self.discount = 0.01 + 0.98 * random.random()


    def configure_crossover(self, genome1, genome2, config):
        super(QGenome, self).configure_crossover(genome1, genome2, config)
        self.discount = random.choice((genome1.discount, genome2.discount))


    def mutate(self, config):
        super(QGenome, self).mutate(config)
        self.discount += random.gauss(0.0, 0.05)
        self.discount = max(0.01, min(0.99, self.discount))


    def distance(self, other, config):
        distance = super(QGenome, self).distance(other, config)
        discount_delta = abs(self.discount - other.discount)
        return distance + discount_delta


def evaluate(env, network, num_episodes):
    total_score = 0

    for _ in range(num_episodes):
            observation = env.reset()
            game_over = False
            score = 0

            while not game_over:
                if random.random() < 0.2:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(network.activate(observation.flatten()))

                observation, reward, done, info = env.perform(action)
                score += reward
                total_score += reward
                game_over = done

    return total_score / num_episodes


class NeatQNetwork(DeepQNetwork):

    def __init__(self, env, elite_size=3, gen_evolve=1, test_episodes=1, verbose=False):
        # type: (None, float, int, int) -> NeatQNetwork
        super(NeatQNetwork, self).__init__()
        self.env = env
        self.gen_evolve = gen_evolve
        self.generation = 0
        self.test_episodes = test_episodes
        self.verbose = verbose
        self.config = None
        self.stats = neat.StatisticsReporter()
        self.population = None
        self.elite_size = elite_size
        self.error_calculator = None
        self.pool = multiprocessing.Pool(4)


    @staticmethod
    def read_config(environment):
        config_path = os.path.join(os.path.dirname(__file__), 'neat_config/neat-config-{}'.format(environment.name))
        return neat.Config(QGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                           neat.DefaultStagnation, config_path)


    def setup(self, environment):
        self.env = environment
        self.config = self.read_config(environment)
        self.error_calculator = ErrorCalculator(environment)
        self.population = neat.Population(self.config)
        self.population.add_reporter(self.stats)

        if self.verbose:
            self.population.add_reporter(neat.StdOutReporter(self.config))


    def fittest_networks(self, n):
        """
        Returns the fittest n networks of the current population.
        :rtype: list
        :param n: The number of fittest networks to create.
        :return: An array of n neural network phenotypes.
        """
        return map(self.phenotype, self.stats.best_unique_genomes(n))


    def phenotype(self, genome):
        """
        Creates and returns the neural network phenotype of the given genome. 
        :param genome: The genome encoding the network.
        """
        return FeedForwardNetwork.create(genome, self.config)


    def evolve(self):
        """
        Runs the genetic algorithm to evolve the population for a specified number of generations.
        """
        print('\n==\nEvolving from generation {0} to {1}.'.format(self.generation, self.generation + self.gen_evolve))
        t0 = time.time()
        best_genome = self.population.run(fitness_function=self.compute_fitness, n=self.gen_evolve)
        self.actor = self.phenotype(best_genome)
        self.generation += self.gen_evolve
        print('Duration: {} seconds.\n==\n'.format(time.time() - t0))


    def compute_fitness(self, population, config):
        # type: ((str, QGenome), neat.Config) -> None
        """
        Computes the fitness of each genome in the population and 
        stores it in their fitness properties.

        :param population: A list of genome_id, genome tuples.
        :param config: The neat config.
        """
        t0 = time.time()
        for _, genome in population:
            genome.fitness = evaluate(self.env, self.phenotype(genome), self.test_episodes)
        print('Synchronous: {}'.format(time.time() - t0))

        t0 = time.time()
        tasks = [self.pool.apply_async(evaluate, (self.env, self.phenotype(genome), self.test_episodes)) for _, genome in population]
        for task, (_, genome) in zip(tasks, population):
            genome.fitness = task.get(timeout=None)
        print('Distributed: {}'.format(time.time() - t0))


    def prepare_for_episode(self):
        self.evolve()


    def did_finish_episode(self, memory):
        pass


    def activate(self, input):
        """
        Feeds the specified `input` vector to the actor network and returns its predicted q vector.
        :param input: A state retrieved from the environment.
        :return: A vector of scalar q values for every action.
        """
        return self.actor.activate(input.flatten())


    def replay(self, experiences):
        pass

