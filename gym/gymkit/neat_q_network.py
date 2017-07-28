from gymkit.q_function_approximator import DeepQNetwork
import neat
import os
from neat.nn import FeedForwardNetwork
import numpy as np
from gymkit.q_agent import QAgent
from gymkit.arena import Arena
from gymkit.environment import Environment
import random
import time
from gymkit.q_models import Experience


def mse(x, y):
    assert len(x) == len(y)
    return np.mean([(pred - target)**2 for pred, target in zip(x, y)])


def compute_fitness(genome, net, episodes, min_reward, max_reward):
    m = int(round(np.log(0.01) / np.log(genome.discount)))
    discount_function = [genome.discount ** (m - i) for i in range(m + 1)]

    reward_errors = []
    for score, experiences in episodes:
        # Compute normalized discounted reward.
        rewards = [e.reward for e in experiences]
        discounted_reward = np.convolve(rewards, discount_function)[m:]
        discounted_reward = 2 * (discounted_reward - min_reward) / (max_reward - min_reward) - 1.0
        discounted_reward = np.clip(discounted_reward, -1.0, 1.0)

        for experience, discounted_reward in zip(experiences, discounted_reward):
            output = net.activate(experience.state.flatten())
            reward_errors.append(float((output[experience.action] - discounted_reward) ** 2))

    return reward_errors


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

    def simulate(self, nets):
        scores = []
        for genome, net in nets:
            state = self.env.reset()
            step = 0
            experiences = []
            while True:
                step += 1
                if random.random() < 0.2:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(net.activate(state.flatten()))

                observation, reward, done, _ = self.env.perform(action)
                experiences.append(Experience(state, action, reward, observation, done))

                state = observation
                if done:
                    break

            score = np.sum([e.reward for e in experiences])
            self.episode_score.append(score)
            scores.append(score)
            self.episode_length.append(step)
            self.test_episodes.append((score, experiences))


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
        for genome, net in nets:
            reward_error = compute_fitness(genome, net, self.test_episodes, self.min_reward, self.max_reward)
            genome.fitness = -np.sum(reward_error) / len(self.test_episodes)
        print("Computing fitness took {0} seconds.\n".format(time.time() - t0))


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
        self.evolve()

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


    # def compute_fitness(self, genomes, config):
    #     """
    #     Computes the fitness of each genome in the population and
    #     assigns it to their fitness properties.
    #     """
    #     for _, genome in genomes:
    #         genome.fitness = self.evaluate(self.phenotype(genome), self.env, self.test_episodes)
    #

    # def evaluate(self, network, env, episodes):
    #     # type: (FeedForwardNetwork, Environment, int) -> float
    #     """
    #     Evaluates the given phenotype on the given number of episodes in the given environment.
    #     :param network: A neural network phenotype.
    #     :param env: An environment to evaluate the phenotype in.
    #     :param episodes: The number of test episodes to run.
    #     :return: The average score achieved in the episodes.
    #     """
    #     agent = QAgent(network, id="test_agent_{}".format(id))
    #     agent.setup(env)
    #     total_score = 0
    #
    #     for _ in range(episodes):
    #         observation = env.reset()
    #         game_over = False
    #         score = 0
    #
    #         while not game_over:
    #             observation, reward, done, info = env.perform(agent.action(observation.flatten()))
    #             score += reward
    #             total_score += reward
    #
    #             if done:
    #                 game_over = True
    #
    #     return total_score / episodes


    def evolve(self):
        print("\n===========\nEvolving from generation {0} to {1}...\n===========\n".format(self.generation, self.generation + self.gen_evolve))
        best_genome = self.population.run(fitness_function=self.error_calculator.evaluate_genomes, n=self.gen_evolve)
        self.actor = self.phenotype(best_genome)
        self.generation += self.gen_evolve


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
