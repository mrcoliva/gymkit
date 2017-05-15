import neat
from neat.nn import FeedForwardNetwork
import os
from agent import Agent
import numpy as np


class NeatAgent(Agent):

    def __init__(self, id='NeatAgent', elite_size=3, verbose=False):
        super(NeatAgent, self).__init__()
        self.env = None
        self.config = None
        self.verbose = verbose
        self.id = id
        self.stats = neat.StatisticsReporter()
        self.population = None
        self.elite_size = elite_size
        self.scores = []
        self.elite_scores = []


    def setup(self, environment):
        self.env = environment
        self.config = self.read_config(environment)
        self.population = neat.Population(self.config)
        self.population.add_reporter(self.stats)

        if self.verbose:
            self.population.add_reporter(neat.StdOutReporter(self.config))


    def read_config(self, environment):
        config_path = os.path.join(os.path.dirname(__file__), 'neat_config/neat-config-{}'.format(environment.name))
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                           neat.DefaultStagnation, config_path)


    def log_episode(self, score):
        self.scores.append(score)


    def fittest_networks(self, n, stats, config):
        return map(lambda genome: FeedForwardNetwork.create(genome, config), stats.best_unique_genomes(n))


    def compute_fitness(self, population, config):
        """
        Computes the fitness of each genome in the population and 
        stores it in their fitness properties.
        
        :param population: A list of genome_id, genome tuples.
        :param config: The neat config.
        """
        genome_network_pairs = map(lambda (_, genome): (genome, FeedForwardNetwork.create(genome, config)), population)

        test_episodes = 1
        for genome, network in genome_network_pairs:
            avg_scores = self.average_scores(self.env, network, test_episodes)
            genome.fitness = self.fitness(avg_scores)


    def fitness(self, average_score):
        # TODO: implement a less naive function.
        """
        :param average_score: The average of the scores reached in the simulated games.
        :return: The fitness of the genome.
        """
        # print average_score
        return average_score / 300


    def average_scores(self, env, network, episodes):
        total_score = 0

        for _ in range(episodes):
            observation = env.reset()
            game_over = False
            score = 0

            while not game_over:
                observation, reward, done, info = env.perform(self.action([network], observation))
                score += reward
                total_score += reward

                if done or score >= 300:
                    game_over = True
                    self.log_episode(score)

        return total_score / episodes


    def evolve(self, networks, generations=1):
        _ = self.population.run(fitness_function=self.compute_fitness, n=generations)
        best_genomes = self.stats.best_genomes(networks)
        return map(lambda genome: FeedForwardNetwork.create(genome, self.config), best_genomes), best_genomes


    def action(self, networks, observation):
        votes = map(lambda network: network.activate(observation), networks)
        return map(np.mean, zip(*votes))


    def run_episode(self, render=False):
        networks, genomes = self.evolve(self.elite_size)
        episode_reward = 0
        observation = self.env.reset()

        while True:
            action = self.action(networks, observation)
            observation, reward, done, _ = self.env.perform(action)
            episode_reward += reward

            if render:
                self.env.render()

            if done:
                return episode_reward
